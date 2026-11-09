#include "simulation.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <exception>
#include <memory>

using namespace godot;

namespace {
Vector3 vector(tennis::Vec3 v) { return Vector3(v.x, v.y, v.z); }
PackedFloat64Array joints(const tennis::JointValues& values) {
    PackedFloat64Array result;
    result.resize(values.size());
    for (std::size_t i=0; i<values.size(); ++i) result.set(i, values[i]);
    return result;
}
tennis::Vec3 physical(Vector3 v) { return {v.x, v.y, v.z}; }
Dictionary poses(const std::vector<tennis::BodyPose>& source) {
    Dictionary result;
    for (const auto& body:source) {
        const auto q=body.orientation;
        result[String(body.name.c_str())]=Transform3D(Basis(Quaternion(q.x,q.y,q.z,q.w)),vector(body.position));
    }
    return result;
}
}

// The boundary converts owned SI/Y-up snapshots into Godot values. Gameplay
// never borrows MuJoCo memory or manipulates solver indices or contact forces.
class TennisBridge : public RefCounted {
    GDCLASS(TennisBridge, RefCounted)
    std::unique_ptr<tennis::Simulation> simulation_;
    tennis::CapturedState checkpoint_;
    String error_;

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("configure", "xml"), &TennisBridge::configure);
        ClassDB::bind_method(D_METHOD("advance", "ticks"), &TennisBridge::advance);
        ClassDB::bind_method(D_METHOD("submit_intent", "time", "targets", "confidence"), &TennisBridge::submit_intent);
        ClassDB::bind_method(D_METHOD("submit_player_intent", "player", "time", "targets", "confidence", "movement_velocity"), &TennisBridge::submit_player_intent);
        ClassDB::bind_method(D_METHOD("feed", "position", "velocity", "spin"), &TennisBridge::feed);
        ClassDB::bind_method(D_METHOD("reset"), &TennisBridge::reset);
        ClassDB::bind_method(D_METHOD("snapshot"), &TennisBridge::snapshot);
        ClassDB::bind_method(D_METHOD("save_checkpoint"), &TennisBridge::save_checkpoint);
        ClassDB::bind_method(D_METHOD("restore_checkpoint"), &TennisBridge::restore_checkpoint);
        ClassDB::bind_method(D_METHOD("reset_prediction", "player", "source"), &TennisBridge::reset_prediction);
        ClassDB::bind_method(D_METHOD("last_error"), &TennisBridge::last_error);
        ClassDB::bind_method(D_METHOD("configuration"), &TennisBridge::configuration);
    }

public:
    bool configure(const String& xml) {
        try {
            auto next = std::make_unique<tennis::Simulation>(xml.utf8().get_data());
            simulation_ = std::move(next);
            checkpoint_ = {};
            error_ = "";
            return true;
        } catch (const std::exception& e) {
            error_ = e.what();
            return false;
        }
    }
    bool advance(int64_t ticks) {
        if (!simulation_ || ticks < 0 || ticks > 20000) {
            error_ = "Advance requires a configured simulation and 0–20000 ticks.";
            return false;
        }
        try {
            simulation_->step(static_cast<uint64_t>(ticks));
            return true;
        } catch (const std::exception& e) { error_ = e.what(); return false; }
    }
    bool submit_intent(double time, const PackedFloat64Array& targets, double confidence) {
        return submit_player_intent(0,time,targets,confidence,Vector3());
    }
    bool submit_player_intent(int64_t player, double time, const PackedFloat64Array& targets,
                             double confidence, Vector3 movement_velocity) {
        if (!simulation_ || targets.is_empty() || targets.size()>128 || player<0 || player>1) {
            error_ = "Intent requires a configured simulation, player 0/1 and model-ordered joint targets.";
            return false;
        }
        tennis::JointValues values(targets.size());
        for (std::size_t i=0; i<values.size(); ++i) values[i] = targets[i];
        const bool accepted = simulation_->set_intent({time, values, confidence,physical(movement_velocity)},static_cast<int>(player));
        error_ = accepted ? "" : "Intent rejected: absent player, invalid planar input, uncertain, future or stale observation.";
        return accepted;
    }
    bool feed(Vector3 position, Vector3 velocity, Vector3 spin) {
        if (!simulation_) return false;
        try {
            simulation_->feed_ball(physical(position), physical(velocity), physical(spin));
            return true;
        } catch (const std::exception& e) { error_ = e.what(); return false; }
    }
    void reset() { if (simulation_) simulation_->reset(); }
    bool save_checkpoint() {
        if (!simulation_) return false;
        checkpoint_ = simulation_->capture();
        return true;
    }
    bool restore_checkpoint() {
        if (!simulation_ || !checkpoint_.state) return false;
        try { simulation_->restore(checkpoint_); return true; }
        catch (const std::exception& e) { error_ = e.what(); return false; }
    }
    String last_error() const { return error_; }
    bool reset_prediction(int64_t player, const Dictionary& source) {
        if(!simulation_ || player<0 || player>1) return false;
        for(const char* key:{"base_position","base_velocity"})
            if(!source.has(key) || source[key].get_type()!=Variant::VECTOR3) return false;
        tennis::PlayerSnapshot state;
        state.base_position=physical(source["base_position"]);
        state.base_velocity=physical(source["base_velocity"]);
        tennis::JointValues* values[]{&state.joint_angles,&state.joint_velocities,&state.controller_targets};
        const char* names[]{"joint_angles","joint_velocities","controller_targets"};
        for(int i=0;i<3;++i) {
            if(!source.has(names[i]) || source[names[i]].get_type()!=Variant::PACKED_FLOAT64_ARRAY) return false;
            const PackedFloat64Array data=source[names[i]];
            if(data.size()>128) return false;
            values[i]->assign(data.ptr(),data.ptr()+data.size());
        }
        try { simulation_->reset_prediction(static_cast<int>(player),state); error_=""; return true; }
        catch(const std::exception& e) { error_=e.what(); return false; }
    }
    String configuration() const { return simulation_ ? String(simulation_->configuration().c_str()) : String(); }
    Dictionary snapshot() const {
        Dictionary result;
        if (!simulation_) return result;
        const auto s = simulation_->snapshot();
        result["tick"] = static_cast<int64_t>(s.tick);
        result["time"] = s.time;
        result["timestep"] = s.timestep;
        result["bodies"] = poses(s.bodies);
        result["ball_velocity"] = vector(s.ball_velocity);
        result["ball_spin"] = vector(s.ball_spin);
        result["racket_velocity"] = vector(s.racket_velocity);
        result["racket_spin"] = vector(s.racket_spin);
        result["joint_angles"] = joints(s.joint_angles);
        result["actuator_efforts"] = joints(s.actuator_efforts);
        result["controller_targets"] = joints(s.controller_targets);
        Array players;
        for (const auto& source:s.players) {
            Dictionary player;
            player["bodies"]=poses(source.bodies);
            Array names;
            for (const auto& name:source.joint_names) names.append(String(name.c_str()));
            player["joint_names"]=names;
            player["joint_angles"]=joints(source.joint_angles);
            player["joint_velocities"]=joints(source.joint_velocities);
            player["base_position"]=vector(source.base_position);
            player["base_velocity"]=vector(source.base_velocity);
            player["actuator_efforts"]=joints(source.actuator_efforts);
            player["controller_targets"]=joints(source.controller_targets);
            player["torso_position"]=vector(source.torso_position);
            player["torso_velocity"]=vector(source.torso_velocity);
            player["movement_effort"]=vector(source.movement_effort);
            player["movement_target"]=vector(source.movement_target);
            player["racket_velocity"]=vector(source.racket_velocity);
            player["racket_spin"]=vector(source.racket_spin);
            player["racket_episodes"]=static_cast<int64_t>(source.racket_episodes);
            player["input_expired"]=source.input_expired;
            player["movable"]=source.movable;
            players.append(player);
        }
        result["players"]=players;
        Dictionary d;
        d["racket_episodes"] = static_cast<int64_t>(s.diagnostics.racket_episodes);
        d["court_episodes"] = static_cast<int64_t>(s.diagnostics.court_episodes);
        d["net_episodes"] = static_cast<int64_t>(s.diagnostics.net_episodes);
        d["target_clipped_ticks"] = static_cast<int64_t>(s.diagnostics.target_clipped_ticks);
        d["torque_saturated_ticks"] = static_cast<int64_t>(s.diagnostics.torque_saturated_ticks);
        d["rate_limited_ticks"] = static_cast<int64_t>(s.diagnostics.rate_limited_ticks);
        d["movement_target_clipped_ticks"] = static_cast<int64_t>(s.diagnostics.movement_target_clipped_ticks);
        d["movement_force_saturated_ticks"] = static_cast<int64_t>(s.diagnostics.movement_force_saturated_ticks);
        d["rejected_inputs"] = static_cast<int64_t>(s.diagnostics.rejected_inputs);
        d["stale_inputs"] = static_cast<int64_t>(s.diagnostics.stale_inputs);
        d["warnings"] = static_cast<int64_t>(s.diagnostics.warnings);
        d["kinetic_energy"] = s.diagnostics.kinetic_energy;
        d["potential_energy"] = s.diagnostics.potential_energy;
        d["actuator_work"] = s.diagnostics.actuator_work;
        d["max_penetration"] = s.diagnostics.max_penetration;
        d["max_racket_speed"] = s.diagnostics.max_racket_speed;
        d["max_stringbed_speed"] = s.diagnostics.max_stringbed_speed;
        d["input_expired"] = s.diagnostics.input_expired;
        d["faulted"] = s.diagnostics.faulted;
        result["diagnostics"] = d;
        Array events;
        for (const auto& c : s.contacts) {
            Dictionary e;
            e["surface"] = String(c.surface.c_str());
            e["player"] = c.player;
            e["position"] = vector(c.position);
            e["first_tick"] = static_cast<int64_t>(c.first_tick);
            e["last_tick"] = static_cast<int64_t>(c.last_tick);
            e["impulse"] = vector(c.impulse);
            e["max_penetration"] = c.max_penetration;
            e["duration"] = c.duration;
            e["incoming_velocity"] = vector(c.incoming_velocity);
            e["outgoing_velocity"] = vector(c.outgoing_velocity);
            e["incoming_spin"] = vector(c.incoming_spin);
            e["outgoing_spin"] = vector(c.outgoing_spin);
            e["complete"] = c.complete;
            events.append(e);
        }
        result["contacts"] = events;
        return result;
    }
};

void initialize_tennis(ModuleInitializationLevel level) {
    if (level == MODULE_INITIALIZATION_LEVEL_SCENE) ClassDB::register_class<TennisBridge>();
}
void uninitialize_tennis(ModuleInitializationLevel) {}

extern "C" GDExtensionBool GDE_EXPORT tennis_library_init(
    GDExtensionInterfaceGetProcAddress address, GDExtensionClassLibraryPtr library,
    GDExtensionInitialization* initialization) {
    GDExtensionBinding::InitObject init(address, library, initialization);
    init.register_initializer(initialize_tennis);
    init.register_terminator(uninitialize_tennis);
    init.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}
