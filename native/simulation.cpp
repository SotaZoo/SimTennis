#include "simulation.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <utility>

namespace tennis {
namespace {
constexpr double input_lifetime = .25;
constexpr std::size_t event_capacity = 32;
constexpr std::array<const char*, joint_count> model_joints{
    "shoulder_yaw", "shoulder_pitch", "elbow", "shoulder_axial", "forearm_roll", "wrist_bend"};
constexpr std::array<const char*, joint_count> motor_names{
    "shoulder_yaw_motor", "shoulder_pitch_motor", "elbow_motor",
    "shoulder_axial_motor", "forearm_roll_motor", "wrist_bend_motor"};
// Engineering controller settings; no physiological strength claim. Bounds and
// effort caps belong to the model; see research/BATCH3_ARM.md for provenance.
const JointValues stiffness{95, 105, 55, 45, 18, 20};
const JointValues damping{7.5, 8, 3.5, 3, 1.2, 1.2};
const JointValues target_rates{3, 3, 4, 4, 5, 4};
// Provisional assisted-support settings, not measurements of human locomotion.
constexpr double movement_speed = 5;
constexpr double movement_force = 700;
constexpr double movement_gain = 350;
constexpr double boundary_deceleration = 4;
constexpr std::array<const char*,5> player_bodies{"torso","upper_arm","forearm","hand","racket"};

Vec3 vec(const mjtNum* value) { return {value[0], value[1], value[2]}; }
bool finite(Vec3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
double norm(Vec3 value) { return std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z); }
void add(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
std::uint64_t xml_hash(const std::string& text) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
    return hash;
}
struct PendingContact {
    bool active = false;
    ContactEvent event;
};
struct PlayerRuntime {
    bool has_intent = false;
    bool direct_effort = false;
    bool input_expired = true;
    Intent intent;
    Effort effort;
    JointValues last_effort{};
    JointValues controller_targets{};
    Vec3 movement_effort;
    Vec3 movement_target;
    std::uint64_t racket_episodes = 0;
};
struct RuntimeState {
    std::uint64_t tick = 0;
    Diagnostics diagnostics;
    // Racket 0, court, net, racket 1. Racket episodes never merge across owners.
    std::array<PendingContact, 4> pending;
    std::vector<ContactEvent> finished;
    std::vector<PlayerRuntime> players;
};
struct PlayerModel {
    std::vector<std::string> names, body_names;
    std::vector<int> joints, motors, bodies;
    JointValues kp, kd, rates;
    std::array<int,2> movement_joints{}, movement_motors{};
    int racket = -1, stringbed = -1;
    bool movable = false;
    JointValues neutral_targets{};
};

bool torque_motor(const mjModel* model, int motor, int joint) {
    return model->actuator_trntype[motor]==mjTRN_JOINT
        && model->actuator_trnid[2*motor]==joint && model->actuator_gear[6*motor]==1
        && model->actuator_ctrllimited[motor]
        && std::isfinite(model->actuator_ctrlrange[2*motor])
        && std::isfinite(model->actuator_ctrlrange[2*motor+1])
        && model->actuator_ctrlrange[2*motor]<model->actuator_ctrlrange[2*motor+1]
        && !model->actuator_forcelimited[motor] && !model->jnt_actfrclimited[joint]
        && model->actuator_dyntype[motor]==mjDYN_NONE
        && model->actuator_gaintype[motor]==mjGAIN_FIXED && model->actuator_gainprm[mjNGAIN*motor]==1
        && model->actuator_biastype[motor]==mjBIAS_NONE;
}
} // namespace

namespace detail {
struct SavedState {
    std::uint64_t model_hash = 0;
    std::vector<mjtNum> integration;
    RuntimeState runtime;
    std::array<mjWarningStat, mjNWARNING> warnings;
};
} // namespace detail

struct Simulation::Impl {
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr, mj_deleteModel};
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData};
    std::uint64_t model_hash = 0;
    int ball = -1, ball_joint = -1;
    std::vector<PlayerModel> players;
    RuntimeState state;

    explicit Impl(const std::string& xml) : model_hash(xml_hash(xml)) {
        mjVFS files;
        mj_defaultVFS(&files);
        if (mj_addBufferVFS(&files, "scenario.xml", xml.data(), static_cast<int>(xml.size())) != 0) {
            mj_deleteVFS(&files);
            throw std::runtime_error("Could not add MuJoCo model to in-memory filesystem");
        }
        char error[2048]{};
        model.reset(mj_loadXML("scenario.xml", &files, error, sizeof(error)));
        mj_deleteVFS(&files);
        if (!model) throw std::invalid_argument(std::string("Invalid MuJoCo model: ") + error);
        if (model->opt.integrator == mjINT_RK4) {
            throw std::invalid_argument("Supported controllers require a single-step integrator; use implicitfast");
        }
        data.reset(mj_makeData(model.get()));
        if (!data) throw std::runtime_error("Could not allocate MuJoCo integration state");
        ball = mj_name2id(model.get(), mjOBJ_BODY, "ball");
        ball_joint = mj_name2id(model.get(), mjOBJ_JOINT, "ball_free");
        if (ball < 0 || ball_joint < 0 || model->jnt_type[ball_joint] != mjJNT_FREE) {
            throw std::invalid_argument("Scenario must contain ball body and ball_free free joint");
        }
        const int count = mj_name2id(model.get(),mjOBJ_BODY,"player1_racket") >= 0 ? 2 : 1;
        if (mj_name2id(model.get(),mjOBJ_BODY,"player2_racket") >= 0)
            throw std::invalid_argument("A shared world supports at most two players");
        for (int index=0;index<count;++index) {
            const std::string prefix=index ? "player1_" : "";
            const auto id = [&](mjtObj type,const char* name) {return mj_name2id(model.get(),type,(prefix+name).c_str());};
            PlayerModel player;
            player.racket=id(mjOBJ_BODY,"racket");
            player.stringbed=id(mjOBJ_GEOM,"stringbed");
            // A small model-owned scalar-joint contract. The reference drill
            // retains its historical names/settings when no descriptor exists.
            const int descriptor=mj_name2id(model.get(),mjOBJ_TEXT,"controller_joints");
            const bool has_settings=mj_name2id(model.get(),mjOBJ_TEXT,"player_bodies")>=0
                || mj_name2id(model.get(),mjOBJ_NUMERIC,"controller_kp")>=0
                || mj_name2id(model.get(),mjOBJ_NUMERIC,"controller_kd")>=0
                || mj_name2id(model.get(),mjOBJ_NUMERIC,"controller_rates")>=0;
            if (descriptor<0 && has_settings)
                throw std::invalid_argument("A model controller description must include controller_joints");
            auto words = [&](const char* name) {
                const int item=mj_name2id(model.get(),mjOBJ_TEXT,name);
                if (item<0) throw std::invalid_argument(std::string("Missing model text: ")+name);
                std::istringstream stream(model->text_data+model->text_adr[item]);
                std::vector<std::string> result;
                for (std::string word;stream>>word;) result.push_back(word);
                if (result.empty() || result.size()>128 || std::set<std::string>(result.begin(),result.end()).size()!=result.size())
                    throw std::invalid_argument("Model axis/body names must be nonempty, unique and bounded");
                return result;
            };
            auto settings = [&](const char* name) {
                const int item=mj_name2id(model.get(),mjOBJ_NUMERIC,name);
                if (item<0 || model->numeric_size[item]!=static_cast<int>(player.names.size()))
                    throw std::invalid_argument(std::string("Invalid controller setting size: ")+name);
                const auto* begin=model->numeric_data+model->numeric_adr[item];
                JointValues result(begin,begin+player.names.size());
                if (!std::all_of(result.begin(),result.end(),[](double v){return std::isfinite(v) && v>0;}))
                    throw std::invalid_argument("Controller gains and rates must be finite and positive");
                return result;
            };
            if (descriptor>=0) {
                player.names=words("controller_joints");
                player.body_names=words("player_bodies");
                player.kp=settings("controller_kp"); player.kd=settings("controller_kd");
                player.rates=settings("controller_rates");
            } else {
                player.names.assign(joint_names.begin(),joint_names.end());
                player.body_names.assign(player_bodies.begin(),player_bodies.end());
                player.kp=stiffness; player.kd=damping; player.rates=target_rates;
            }
            for (const auto& name:player.body_names) {
                const int body=id(mjOBJ_BODY,name.c_str());
                if (descriptor>=0 && body<0) throw std::invalid_argument("Missing declared player body: "+name);
                player.bodies.push_back(body);
            }
            for (std::size_t i=0; i<player.names.size(); ++i) {
                const std::string name=descriptor>=0 ? player.names[i] : model_joints[i];
                const std::string motor_name=descriptor>=0 ? name+"_motor" : motor_names[i];
                const int joint=id(mjOBJ_JOINT,name.c_str()),motor=id(mjOBJ_ACTUATOR,motor_name.c_str());
                if ((joint<0)!=(motor<0) || (descriptor>=0 && joint<0))
                    throw std::invalid_argument("Controller joint and motor must be supplied together: "+name);
                if (joint>=0 && (model->jnt_type[joint]!=mjJNT_HINGE || !torque_motor(model.get(),motor,joint)))
                    throw std::invalid_argument("Controller axes require bounded unit-gear hinge torque motors");
                player.joints.push_back(joint); player.motors.push_back(motor);
            }
            player.neutral_targets.resize(player.names.size());
            player.movement_joints={id(mjOBJ_JOINT,"movement_x"),id(mjOBJ_JOINT,"movement_z")};
            player.movement_motors={id(mjOBJ_ACTUATOR,"movement_x_motor"),id(mjOBJ_ACTUATOR,"movement_z_motor")};
            player.movable=player.movement_joints[0]>=0;
            for (int axis=0;axis<2;++axis) {
                const int joint=player.movement_joints[axis],motor=player.movement_motors[axis];
                if ((joint>=0)!=player.movable || (motor>=0)!=player.movable)
                    throw std::invalid_argument("Planar support requires both slide joints and both motors");
                if (player.movable && (model->jnt_type[joint]!=mjJNT_SLIDE || !model->jnt_limited[joint]
                    || !torque_motor(model.get(),motor,joint)))
                    throw std::invalid_argument("Planar support requires bounded slides with unit-gear joint motors");
                if (descriptor>=0 && player.movable && model->jnt_bodyid[joint]!=player.bodies.front())
                    throw std::invalid_argument("The first player_bodies entry must own the planar support joints");
            }
            players.push_back(player);
        }
    }

    Vec3 ball_velocity() const {
        const auto* velocity = data->qvel + model->jnt_dofadr[ball_joint];
        return vec(velocity);
    }
    Vec3 ball_spin() const {
        const auto* velocity = data->qvel + model->jnt_dofadr[ball_joint];
        const auto* quaternion = data->qpos + model->jnt_qposadr[ball_joint] + 3;
        mjtNum world[3];
        mju_rotVecQuat(world, velocity+3, quaternion);
        return vec(world);
    }

    void control() {
        bool target_clipped = false, torque_saturated = false, rate_limited = false;
        bool movement_clipped = false, movement_saturated = false;
        for (std::size_t index=0;index<players.size();++index) {
            const auto& player=players[index];
            auto& runtime=state.players[index];
            const double command_time = runtime.direct_effort ? runtime.effort.time : runtime.intent.time;
            runtime.input_expired = !runtime.has_intent || data->time-command_time > input_lifetime;
            for (std::size_t i=0; i<player.joints.size(); ++i) {
                if (player.joints[i] < 0) continue;
                if (runtime.direct_effort) {
                    const int motor = player.motors[i];
                    const double requested = runtime.input_expired ? 0 : runtime.effort.torques[i];
                    const double applied = std::clamp(requested, model->actuator_ctrlrange[2*motor], model->actuator_ctrlrange[2*motor+1]);
                    torque_saturated = torque_saturated || applied != requested;
                    data->ctrl[motor] = applied;
                    runtime.last_effort[i] = applied;
                    continue;
                }
                const int q = model->jnt_qposadr[player.joints[i]];
                const int v = model->jnt_dofadr[player.joints[i]];
                double target = runtime.input_expired ? player.neutral_targets[i] : runtime.intent.targets[i];
                if (model->jnt_limited[player.joints[i]]) {
                    const double bounded = std::clamp(target, model->jnt_range[2*player.joints[i]], model->jnt_range[2*player.joints[i]+1]);
                    target_clipped = target_clipped || target != bounded;
                    target = bounded;
                }
                // Slew the setpoint, never qpos/qvel. Ordinary bounded differences
                // cannot jump across an excluded angular interval at +/-pi.
                const double delta = target-runtime.controller_targets[i];
                const double travel = player.rates[i]*model->opt.timestep;
                const double bounded_delta = std::clamp(delta, -travel, travel);
                rate_limited = rate_limited || delta != bounded_delta;
                runtime.controller_targets[i] += bounded_delta;
                // Gravity/Coriolis compensation obeys the same finite motor caps.
                const double requested = player.kp[i]*(runtime.controller_targets[i]-data->qpos[q])
                    - player.kd[i]*data->qvel[v] + data->qfrc_bias[v];
                double applied = requested;
                if (model->actuator_ctrllimited[player.motors[i]]) {
                    applied = std::clamp(requested, model->actuator_ctrlrange[2*player.motors[i]], model->actuator_ctrlrange[2*player.motors[i]+1]);
                }
                torque_saturated = torque_saturated || applied != requested;
                data->ctrl[player.motors[i]] = applied;
                runtime.last_effort[i] = applied;
            }
            if (!player.movable) continue;
            if (runtime.direct_effort) {
                const Vec3 requested = runtime.input_expired ? Vec3{} : runtime.effort.movement_force;
                const double scale = movement_force/std::max(movement_force, std::hypot(requested.x, requested.z));
                const int x = player.movement_motors[0], z = player.movement_motors[1];
                runtime.movement_effort = {
                    std::clamp(requested.x*scale, model->actuator_ctrlrange[2*x], model->actuator_ctrlrange[2*x+1]), 0,
                    std::clamp(requested.z*scale, model->actuator_ctrlrange[2*z], model->actuator_ctrlrange[2*z+1])};
                movement_saturated = movement_saturated || runtime.movement_effort.x != requested.x || runtime.movement_effort.z != requested.z;
                runtime.movement_target = {};
                data->ctrl[x] = runtime.movement_effort.x;
                data->ctrl[z] = runtime.movement_effort.z;
                continue;
            }
            const auto requested=runtime.input_expired ? Vec3{} : runtime.intent.movement_velocity;
            const double scale=movement_speed/std::max(movement_speed,std::hypot(requested.x,requested.z));
            double desired[2]{requested.x*scale,requested.z*scale};
            double force[2]{};
            for (int axis=0;axis<2;++axis) {
                const int joint=player.movement_joints[axis];
                const double q=data->qpos[model->jnt_qposadr[joint]];
                const double distance=desired[axis]>=0 ? model->jnt_range[2*joint+1]-q : q-model->jnt_range[2*joint];
                // Brake before the finite soft stop. Only the requested velocity
                // changes; the solver remains responsible for physical motion.
                const double safe_speed=std::sqrt(2*boundary_deceleration*std::max(0.,distance-.03));
                desired[axis]=std::clamp(desired[axis],-safe_speed,safe_speed);
                force[axis]=movement_gain*(desired[axis]-data->qvel[model->jnt_dofadr[joint]]);
            }
            runtime.movement_target={desired[0],0,desired[1]};
            movement_clipped=movement_clipped || desired[0]!=requested.x || desired[1]!=requested.z;
            const double force_scale=movement_force/std::max(movement_force,std::hypot(force[0],force[1]));
            movement_saturated=movement_saturated || force_scale<1;
            runtime.movement_effort={force[0]*force_scale,0,force[1]*force_scale};
            data->ctrl[player.movement_motors[0]]=runtime.movement_effort.x;
            data->ctrl[player.movement_motors[1]]=runtime.movement_effort.z;
        }
        state.diagnostics.input_expired=state.players.front().input_expired;
        if (target_clipped) ++state.diagnostics.target_clipped_ticks;
        if (torque_saturated) ++state.diagnostics.torque_saturated_ticks;
        if (rate_limited) ++state.diagnostics.rate_limited_ticks;
        if (movement_clipped) ++state.diagnostics.movement_target_clipped_ticks;
        if (movement_saturated) ++state.diagnostics.movement_force_saturated_ticks;
    }

    void refresh_presentation_state() {
        // Position/velocity caches only: rendering must never solve constraints
        // or alter warm-start state. Batch boundaries therefore cannot affect
        // future physics. mj_step1 recomputes dynamics before the next control.
        mj_kinematics(model.get(), data.get());
        mj_comPos(model.get(), data.get());
        mj_makeM(model.get(), data.get());
        mj_comVel(model.get(), data.get());
    }

    void sample_racket_speed() {
        mjtNum velocity[6];
        for (const auto& player:players) {
            if (player.racket >= 0) {
                mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, player.racket, velocity, 0);
                state.diagnostics.max_racket_speed = std::max(state.diagnostics.max_racket_speed, norm(vec(velocity+3)));
            }
            if (player.stringbed >= 0) {
                mj_objectVelocity(model.get(), data.get(), mjOBJ_GEOM, player.stringbed, velocity, 0);
                state.diagnostics.max_stringbed_speed = std::max(state.diagnostics.max_stringbed_speed, norm(vec(velocity+3)));
            }
        }
    }

    int surface(int geometry) const {
        const char* name = mj_id2name(model.get(), mjOBJ_GEOM, geometry);
        if (!name) return -1;
        const std::string item(name);
        if (item == "stringbed" || item.rfind("racket_",0) == 0) return 0;
        if (item == "player1_stringbed" || item.rfind("player1_racket_",0) == 0) return 3;
        if (item == "court") return 1;
        if (item.rfind("net",0) == 0) return 2;
        return -1;
    }

    void collect_contacts(Vec3 incoming_position, Vec3 incoming_velocity, Vec3 incoming_spin) {
        std::array<bool,4> touching{};
        std::array<Vec3,4> impulse{};
        std::array<double,4> penetration{};
        // mj_step2 has solved these contacts but has not refreshed them for the
        // new qpos. Capture their forces before mj_forward overwrites anything.
        for (int i=0; i<data->ncon; ++i) {
            const auto& contact = data->contact[i];
            const bool ball_first = model->geom_bodyid[contact.geom[0]] == ball;
            const bool ball_second = model->geom_bodyid[contact.geom[1]] == ball;
            if (!ball_first && !ball_second) continue;
            const int index = surface(contact.geom[ball_first ? 1 : 0]);
            if (index < 0) continue;
            mjtNum wrench[6]{}, world[3]{};
            mj_contactForce(model.get(), data.get(), i, wrench);
            if (wrench[0] <= 1e-10) continue;
            mju_mulMatTVec(world, contact.frame, wrench, 3, 3);
            const double scale = (ball_first ? -1 : 1) * model->opt.timestep;
            add(impulse[index], {world[0]*scale,world[1]*scale,world[2]*scale});
            touching[index] = true;
            penetration[index] = std::max(penetration[index], -contact.dist);
        }
        constexpr std::array<const char*,4> names{"racket","court","net","racket"};
        for (std::size_t i=0; i<4; ++i) {
            auto& pending = state.pending[i];
            if (touching[i]) {
                if (!pending.active) {
                    pending.active = true;
                    pending.event = {};
                    pending.event.surface = names[i];
                    pending.event.player = i==0 ? 0 : (i==3 ? 1 : -1);
                    pending.event.first_tick = state.tick-1;
                    pending.event.position = incoming_position;
                    pending.event.incoming_velocity = incoming_velocity;
                    pending.event.incoming_spin = incoming_spin;
                    if (i == 0 || i == 3) {
                        ++state.diagnostics.racket_episodes;
                        ++state.players[pending.event.player].racket_episodes;
                    }
                    if (i == 1) ++state.diagnostics.court_episodes;
                    if (i == 2) ++state.diagnostics.net_episodes;
                }
                pending.event.last_tick = state.tick;
                pending.event.duration += model->opt.timestep;
                pending.event.max_penetration = std::max(pending.event.max_penetration, penetration[i]);
                state.diagnostics.max_penetration = std::max(state.diagnostics.max_penetration, penetration[i]);
                add(pending.event.impulse, impulse[i]);
                pending.event.outgoing_velocity = ball_velocity();
                pending.event.outgoing_spin = ball_spin();
            } else if (pending.active) {
                pending.event.complete = true;
                state.finished.push_back(pending.event);
                if (state.finished.size() > event_capacity) state.finished.erase(state.finished.begin());
                pending = {};
            }
        }
    }
};

Simulation::Simulation(const std::string& model_xml) : impl_(std::make_unique<Impl>(model_xml)) { reset(); }
Simulation::~Simulation() = default;

void Simulation::reset() {
    auto& p = *impl_;
    const int ready = mj_name2id(p.model.get(), mjOBJ_KEY, "ready");
    if (ready >= 0) mj_resetDataKeyframe(p.model.get(), p.data.get(), ready);
    else mj_resetData(p.model.get(), p.data.get());
    p.state = {};
    p.state.players.resize(p.players.size());
    for (std::size_t index=0;index<p.players.size();++index) {
        auto& player=p.players[index];
        for (std::size_t i=0; i<player.joints.size(); ++i)
            player.neutral_targets[i] = player.joints[i] < 0 ? 0 : p.data->qpos[p.model->jnt_qposadr[player.joints[i]]];
        p.state.players[index].controller_targets=player.neutral_targets;
        p.state.players[index].last_effort.assign(player.joints.size(),0);
    }
    mj_forward(p.model.get(), p.data.get());
    for (const auto& player:p.players) if (player.movable) {
        for (int axis=0;axis<2;++axis) {
            const auto actual=vec(p.data->xaxis+3*player.movement_joints[axis]);
            if (std::abs(actual.x-(axis==0))>1e-10 || std::abs(actual.y)>1e-10 || std::abs(actual.z-(axis==1))>1e-10)
                throw std::invalid_argument("Support slide axes must be positive WORLD X and Z");
        }
    }
}

void Simulation::set_player_start(Vec3 position, int index) {
    auto& p = *impl_;
    if (p.data->time != 0 || p.state.tick != 0 || p.state.diagnostics.faulted
        || index < 0 || static_cast<std::size_t>(index) >= p.players.size()
        || !finite(position))
        throw std::invalid_argument("Player start requires finite reset-time coordinates and an existing player");
    const auto& player = p.players[index];
    const auto origin = snapshot().players[index].base_position;
    if (!player.movable || std::abs(position.y-origin.y) > 1e-10)
        throw std::invalid_argument("Player start requires a planar movable support");
    const double offset[]{position.x-origin.x, position.z-origin.z};
    double targets[2];
    for (int axis=0; axis<2; ++axis) {
        const int joint = player.movement_joints[axis];
        targets[axis] = p.data->qpos[p.model->jnt_qposadr[joint]] + offset[axis];
        if (p.model->jnt_limited[joint]
            && (targets[axis] < p.model->jnt_range[2*joint]
                || targets[axis] > p.model->jnt_range[2*joint+1]))
            throw std::invalid_argument("Player start is outside the support slide limits");
    }
    for (int axis=0; axis<2; ++axis) {
        const int joint = player.movement_joints[axis];
        p.data->qpos[p.model->jnt_qposadr[joint]] = targets[axis];
        p.data->qvel[p.model->jnt_dofadr[joint]] = 0;
    }
    mju_zero(p.data->qacc_warmstart, p.model->nv);
    mj_forward(p.model.get(), p.data.get());
}

void Simulation::feed_ball(Vec3 position, Vec3 velocity, Vec3 spin) {
    if (!finite(position) || !finite(velocity) || !finite(spin)) throw std::invalid_argument("Ball setup must be finite");
    auto& p = *impl_;
    auto* pose = p.data->qpos + p.model->jnt_qposadr[p.ball_joint];
    auto* speed = p.data->qvel + p.model->jnt_dofadr[p.ball_joint];
    pose[0]=position.x; pose[1]=position.y; pose[2]=position.z;
    pose[3]=1; pose[4]=0; pose[5]=0; pose[6]=0;
    speed[0]=velocity.x; speed[1]=velocity.y; speed[2]=velocity.z;
    speed[3]=spin.x; speed[4]=spin.y; speed[5]=spin.z;
    p.state.pending = {};
    p.state.finished.clear();
    p.state.diagnostics.racket_episodes = p.state.diagnostics.court_episodes = p.state.diagnostics.net_episodes = 0;
    for (auto& player:p.state.players) player.racket_episodes=0;
    p.state.diagnostics.max_penetration = 0;
    mju_zero(p.data->qacc_warmstart, p.model->nv);
    mj_forward(p.model.get(), p.data.get());
}

bool Simulation::set_intent(const Intent& intent, int player) {
    auto& p = *impl_;
    if (player<0 || static_cast<std::size_t>(player)>=p.players.size()) {
        ++p.state.diagnostics.rejected_inputs;
        return false;
    }
    auto& runtime=p.state.players[player];
    const bool invalid = !std::isfinite(intent.time) || !std::isfinite(intent.confidence)
        || intent.targets.size()!=p.players[player].joints.size()
        || intent.confidence < .5 || intent.confidence > 1
        || !std::all_of(intent.targets.begin(), intent.targets.end(), [](double angle) { return std::isfinite(angle); })
        || !finite(intent.movement_velocity) || intent.movement_velocity.y!=0
        || intent.time > p.data->time + 1e-9;
    const bool stale = intent.time < p.data->time - input_lifetime
        || (runtime.has_intent && intent.time < (runtime.direct_effort ? runtime.effort.time : runtime.intent.time));
    if (invalid || stale) {
        ++p.state.diagnostics.rejected_inputs;
        if (stale) ++p.state.diagnostics.stale_inputs;
        return false;
    }
    runtime.has_intent = true;
    runtime.direct_effort = false;
    runtime.intent = intent;
    runtime.input_expired = false;
    if (player==0) p.state.diagnostics.input_expired = false;
    return true;
}

bool Simulation::set_effort(const Effort& effort, int player) {
    auto& p = *impl_;
    if (player < 0 || static_cast<std::size_t>(player) >= p.players.size()) {
        ++p.state.diagnostics.rejected_inputs;
        return false;
    }
    auto& runtime = p.state.players[player];
    const bool invalid = !std::isfinite(effort.time) || effort.time > p.data->time+1e-9
        || effort.torques.size() != p.players[player].joints.size()
        || !std::all_of(effort.torques.begin(), effort.torques.end(), [](double v) { return std::isfinite(v); })
        || !finite(effort.movement_force) || effort.movement_force.y != 0
        || (!p.players[player].movable && norm(effort.movement_force) != 0);
    const double previous_time = runtime.direct_effort ? runtime.effort.time : runtime.intent.time;
    const bool stale = effort.time < p.data->time-input_lifetime
        || (runtime.has_intent && effort.time < previous_time);
    if (invalid || stale) {
        ++p.state.diagnostics.rejected_inputs;
        if (stale) ++p.state.diagnostics.stale_inputs;
        return false;
    }
    runtime.has_intent = true;
    runtime.direct_effort = true;
    runtime.effort = effort;
    runtime.input_expired = false;
    if (player == 0) p.state.diagnostics.input_expired = false;
    return true;
}

void Simulation::reset_prediction(int index, const PlayerSnapshot& source) {
    auto& p=*impl_;
    if(index<0 || static_cast<std::size_t>(index)>=p.players.size())
        throw std::invalid_argument("Prediction requires an existing player");
    const auto& player=p.players[index];
    const auto valid_joints=[&](const JointValues& values) {
        return values.size()==player.joints.size() && std::all_of(values.begin(),values.end(),
            [](double x) { return std::isfinite(x) && std::abs(x)<=100; });
    };
    if(!player.movable || !finite(source.base_position) || !finite(source.base_velocity)
        || norm(source.base_position)>100 || norm(source.base_velocity)>100
        || !valid_joints(source.joint_angles) || !valid_joints(source.joint_velocities)
        || !valid_joints(source.controller_targets))
        throw std::invalid_argument("Prediction requires bounded finite player observations");
    reset();
    const auto origin=snapshot().players[index].base_position;
    const double offset[]{source.base_position.x-origin.x,source.base_position.z-origin.z};
    const double velocity[]{source.base_velocity.x,source.base_velocity.z};
    for(int axis=0;axis<2;++axis) {
        const int joint=player.movement_joints[axis];
        p.data->qpos[p.model->jnt_qposadr[joint]]+=offset[axis];
        p.data->qvel[p.model->jnt_dofadr[joint]]=velocity[axis];
    }
    for(std::size_t i=0;i<player.joints.size();++i) {
        const int joint=player.joints[i];
        if(joint<0) continue;
        p.data->qpos[p.model->jnt_qposadr[joint]]=source.joint_angles[i];
        p.data->qvel[p.model->jnt_dofadr[joint]]=source.joint_velocities[i];
    }
    p.state.players[index].controller_targets=source.controller_targets;
    // The predictor cannot create a speculative racket/ball contact. Only the
    // server's ball and contact events are shown and adjudicated.
    feed_ball({100,100,100},{});
}

void Simulation::step(std::uint64_t ticks) {
    auto& p = *impl_;
    if (p.state.diagnostics.faulted) throw std::runtime_error("Simulation is faulted; reset before advancing");
    if (ticks > 2000000) throw std::invalid_argument("A step request is limited to two million physics ticks");
    for (std::uint64_t tick=0; tick<ticks; ++tick) {
        const double old_time = p.data->time;
        constexpr std::array<int,3> numerical_warnings{mjWARN_BADQPOS,mjWARN_BADQVEL,mjWARN_BADQACC};
        std::array<int,3> warning_counts{};
        for (std::size_t i=0; i<3; ++i) warning_counts[i] = p.data->warning[numerical_warnings[i]].number;
        const auto changed_numerical_warning = [&] {
            for (std::size_t i=0; i<3; ++i) if (p.data->warning[numerical_warnings[i]].number != warning_counts[i]) return true;
            return false;
        };
        // MuJoCo's split step computes current kinematics/bias forces before
        // bounded control, without a process-global callback. Both implicit
        // integrators are supported by mj_step2; RK4 is rejected at load time.
        mj_step1(p.model.get(), p.data.get());
        if (p.data->time != old_time || changed_numerical_warning()) {
            p.state.diagnostics.faulted = true;
            throw std::runtime_error("MuJoCo numerical state reset during step1; pause and inspect the model");
        }
        p.sample_racket_speed();
        p.control();
        const auto incoming_position = vec(p.data->qpos+p.model->jnt_qposadr[p.ball_joint]);
        const auto incoming = p.ball_velocity();
        const auto incoming_spin = p.ball_spin();
        double power_before = 0;
        for (std::size_t index=0;index<p.players.size();++index) {
            const auto& player=p.players[index];
            const auto& runtime=p.state.players[index];
            for (std::size_t i=0;i<player.joints.size();++i) if (player.joints[i]>=0)
                power_before+=runtime.last_effort[i]*p.data->qvel[p.model->jnt_dofadr[player.joints[i]]];
            if (player.movable) {
                power_before+=runtime.movement_effort.x*p.data->qvel[p.model->jnt_dofadr[player.movement_joints[0]]];
                power_before+=runtime.movement_effort.z*p.data->qvel[p.model->jnt_dofadr[player.movement_joints[1]]];
            }
        }
        mj_step2(p.model.get(), p.data.get());
        const double expected_time = old_time+p.model->opt.timestep;
        if (!std::isfinite(p.data->time) || p.data->time != expected_time || changed_numerical_warning()) {
            p.state.diagnostics.faulted = true;
            throw std::runtime_error("MuJoCo reset after numerical instability; simulation must be paused and inspected");
        }
        ++p.state.tick;
        p.collect_contacts(incoming_position, incoming, incoming_spin);
        double actuator_power = 0;
        for (int i=0; i<p.model->nv; ++i) actuator_power += p.data->qfrc_actuator[i]*p.data->qvel[i];
        p.state.diagnostics.actuator_work += .5*(power_before+actuator_power)*p.model->opt.timestep;
    }
    // Public transforms are evaluated at the returned time, never one physics
    // tick behind it. The impact data above is already owned by our event log.
    if (ticks) {
        p.refresh_presentation_state();
        p.sample_racket_speed();
    }
}

Snapshot Simulation::snapshot() const {
    const auto& p = *impl_;
    Snapshot result;
    result.tick = p.state.tick;
    result.time = p.data->time;
    result.timestep = p.model->opt.timestep;
    result.diagnostics = p.state.diagnostics;
    mj_energyPos(p.model.get(), p.data.get());
    mj_energyVel(p.model.get(), p.data.get());
    result.diagnostics.potential_energy = p.data->energy[0];
    result.diagnostics.kinetic_energy = p.data->energy[1];
    for (int i=0; i<mjNWARNING; ++i) result.diagnostics.warnings += p.data->warning[i].number;
    for (int i=1; i<p.model->nbody; ++i) {
        const char* name = mj_id2name(p.model.get(), mjOBJ_BODY, i);
        if (!name) continue;
        const auto* quaternion = p.data->xquat + 4*i;
        result.bodies.push_back({name, vec(p.data->xpos+3*i), {quaternion[0],quaternion[1],quaternion[2],quaternion[3]}});
    }
    result.ball_velocity = p.ball_velocity();
    result.ball_spin = p.ball_spin();
    for (std::size_t index=0;index<p.players.size();++index) {
        const auto& model=p.players[index];
        const auto& runtime=p.state.players[index];
        PlayerSnapshot player;
        for (std::size_t i=0;i<model.bodies.size();++i) {
            const int id=model.bodies[i];
            if (id<0) continue;
            const auto* q=p.data->xquat+4*id;
            player.bodies.push_back({model.body_names[i],vec(p.data->xpos+3*id),{q[0],q[1],q[2],q[3]}});
        }
        if (model.bodies[0]>=0) {
            player.torso_position=vec(p.data->xpos+3*model.bodies[0]);
            mjtNum v[6];
            mj_objectVelocity(p.model.get(),p.data.get(),mjOBJ_BODY,model.bodies[0],v,0);
            // The support's orientation is fixed, so COM and origin velocities
            // coincide even if a future support uses a nonzero COM offset.
            player.torso_velocity=vec(v+3);
        }
        player.base_position=player.torso_position;
        player.base_velocity=player.torso_velocity;
        if (model.racket>=0) {
            mjtNum velocity[6];
            mj_objectVelocity(p.model.get(),p.data.get(),mjOBJ_BODY,model.racket,velocity,0);
            player.racket_velocity=vec(velocity+3);player.racket_spin=vec(velocity);
        }
        player.joint_names=model.names;
        player.joint_angles.resize(model.joints.size());
        player.joint_velocities.resize(model.joints.size());
        for (std::size_t i=0;i<model.joints.size();++i) if (model.joints[i]>=0) {
            player.joint_angles[i]=p.data->qpos[p.model->jnt_qposadr[model.joints[i]]];
            player.joint_velocities[i]=p.data->qvel[p.model->jnt_dofadr[model.joints[i]]];
        }
        player.actuator_efforts=runtime.last_effort;
        player.controller_targets=runtime.controller_targets;
        player.movement_effort=runtime.movement_effort;
        player.movement_target=runtime.movement_target;
        player.input_expired=runtime.input_expired;
        player.racket_episodes=runtime.racket_episodes;
        player.movable=model.movable;
        result.players.push_back(std::move(player));
    }
    const auto& primary=result.players.front();
    result.racket_velocity=primary.racket_velocity;
    result.racket_spin=primary.racket_spin;
    result.joint_angles=primary.joint_angles;
    result.actuator_efforts=primary.actuator_efforts;
    result.controller_targets=primary.controller_targets;
    result.contacts = p.state.finished;
    for (const auto& pending : p.state.pending) if (pending.active) result.contacts.push_back(pending.event);
    std::sort(result.contacts.begin(), result.contacts.end(), [](const auto& a, const auto& b) { return a.first_tick < b.first_tick; });
    if (result.contacts.size() > event_capacity) result.contacts.erase(result.contacts.begin(), result.contacts.end()-event_capacity);
    return result;
}

CapturedState Simulation::capture() const {
    const auto& p = *impl_;
    auto saved = std::make_shared<detail::SavedState>();
    saved->model_hash = p.model_hash;
    saved->integration.resize(mj_stateSize(p.model.get(), mjSTATE_INTEGRATION));
    mj_getState(p.model.get(), p.data.get(), saved->integration.data(), mjSTATE_INTEGRATION);
    saved->runtime = p.state;
    std::copy_n(p.data->warning, mjNWARNING, saved->warnings.begin());
    return {std::move(saved)};
}

void Simulation::restore(const CapturedState& captured) {
    auto& p = *impl_;
    if (!captured.state || captured.state->model_hash != p.model_hash
        || captured.state->integration.size() != static_cast<std::size_t>(mj_stateSize(p.model.get(), mjSTATE_INTEGRATION))) {
        throw std::invalid_argument("Checkpoint must belong to the same model/configuration");
    }
    mj_setState(p.model.get(), p.data.get(), captured.state->integration.data(), mjSTATE_INTEGRATION);
    p.state = captured.state->runtime;
    std::copy(captured.state->warnings.begin(), captured.state->warnings.end(), p.data->warning);
    p.refresh_presentation_state();
}

std::string Simulation::configuration() const {
    const auto& p = *impl_;
    std::ostringstream out;
    out << "MuJoCo " << mj_versionString() << "; model_fnv1a64=" << std::hex << p.model_hash << std::dec
        << "; dt=" << p.model->opt.timestep << " s; Y-up SI; integration_state=" << mjSTATE_INTEGRATION
        << "; joints=" << p.model->njnt << "; actuators=" << p.model->nu
        << "; density=" << p.model->opt.density << "; viscosity=" << p.model->opt.viscosity
        << "; integrator=" << p.model->opt.integrator << "; disableflags=" << p.model->opt.disableflags;
    out << "; controller=model-axes-v1";
    for (std::size_t index=0;index<p.players.size();++index) {
        const auto& player=p.players[index];
        if (p.players.size()>1) out<<"; player="<<index;
        for (std::size_t i=0; i<player.joints.size(); ++i) {
            if (player.joints[i] < 0) continue;
            out << "; " << player.names[i] << " range=" << p.model->jnt_range[2*player.joints[i]]
                << ',' << p.model->jnt_range[2*player.joints[i]+1] << " rad; rate=" << player.rates[i]
                << " rad/s; kp=" << player.kp[i] << "; kd=" << player.kd[i]
                << "; effort=" << p.model->actuator_ctrlrange[2*player.motors[i]]
                << ',' << p.model->actuator_ctrlrange[2*player.motors[i]+1] << " Nm";
        }
        if (player.movable) out<<"; assisted_support=v1; movement_speed="<<movement_speed
            <<" m/s; resultant_force="<<movement_force<<" N; velocity_gain="<<movement_gain
            <<" N/(m/s); boundary_deceleration="<<boundary_deceleration<<" m/s2";
    }
    for (int i=0; i<p.model->npair; ++i) {
        out << "; pair" << i << " solref=" << p.model->pair_solref[2*i] << ',' << p.model->pair_solref[2*i+1]
            << " friction=" << p.model->pair_friction[5*i] << ',' << p.model->pair_friction[5*i+1];
    }
    return out.str();
}

} // namespace tennis
