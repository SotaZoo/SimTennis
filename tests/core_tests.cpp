#include "simulation.hpp"
#include "physics_references.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
constexpr double runtime_timestep = .000125;
void check(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
double length(tennis::Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
tennis::Vec3 subtract(tennis::Vec3 a, tennis::Vec3 b) {
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}
std::string read(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path);
    return std::string(std::istreambuf_iterator<char>(file), {});
}
void replace(std::string& text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos == std::string::npos) throw std::runtime_error("Missing fixture token " + from);
    text.replace(pos, from.size(), to);
}
const tennis::BodyPose& body(const tennis::Snapshot& snapshot, const std::string& name) {
    for (const auto& item : snapshot.bodies) if (item.name == name) return item;
    throw std::runtime_error("Missing body " + name);
}

// Fixtures isolate conservation and restitution from actuators, air and gravity.
std::string collision_fixture(bool free_racket, double step = runtime_timestep) {
    std::ostringstream xml;
    xml << "<mujoco><option timestep='" << step
        << "' gravity='0 0 0' integrator='implicitfast' cone='elliptic' iterations='100'>"
        << "<flag energy='enable' refsafe='disable'/></option>"
        << "<worldbody><body name='ball' pos='0 0 -.4'><freejoint name='ball_free'/>"
        << "<inertial pos='0 0 0' mass='.057' diaginertia='.00004138 .00004138 .00004138'/>"
        << "<geom name='ball_geom' type='sphere' size='.033' contype='1' conaffinity='1'/>"
        << "</body><body name='racket' pos='0 0 0'>";
    if (free_racket) xml << "<freejoint name='racket_free'/>";
    xml << "<inertial pos='0 0 0' mass='.32' diaginertia='.015 .018 .004'/>"
        << "<geom name='stringbed' type='box' size='.18 .23 .005'/>"
        << "</body></worldbody><contact><pair geom1='ball_geom' geom2='stringbed' "
        << "condim='3' friction='.7 .7 .001 .0001 .0001' solref='-500000 -50' "
        << "solimp='.95 .95 .001'/></contact></mujoco>";
    return xml.str();
}

void free_flight() {
    std::string xml = collision_fixture(false);
    replace(xml, "gravity='0 0 0'", "gravity='0 -9.81 0'");
    tennis::Simulation sim(xml);
    sim.feed_ball({3, 3, 0}, {2, 1, 0}, {0, 0, 0});
    sim.step(800);
    const auto result = sim.snapshot();
    const auto p = body(result, "ball").position;
    check(std::abs(result.ball_velocity.y - (1 - .981)) < 1e-10, "free flight matches gravity impulse");
    check(std::abs(p.x - 3.2) < 1e-10, "no-air horizontal speed is conserved");
    check(std::abs(p.y - (3 + .1 - .5*9.81*.1*.1)) < .0003, "free flight matches ballistic position within integration error");
    check(result.diagnostics.racket_episodes == 0, "far miss records no racket episode");
    const double expected_ke = .5*.057*length(result.ball_velocity)*length(result.ball_velocity);
    check(std::abs(result.diagnostics.kinetic_energy-expected_ke) < 1e-12,
          "reported kinetic energy agrees with independent rigid-ball energy");
}

tennis::Snapshot impact(bool free_racket, tennis::Vec3 position, tennis::Vec3 velocity,
                        tennis::Vec3 spin = {}, double dt = runtime_timestep) {
    tennis::Simulation sim(collision_fixture(free_racket, dt));
    sim.feed_ball(position, velocity, spin);
    sim.step(static_cast<std::uint64_t>(std::llround(.12 / dt)));
    return sim.snapshot();
}

void contact_behaviors() {
    const auto held = impact(false, {0,0,-.4}, {0,0,10});
    check(held.ball_velocity.z < 0, "clamped face actually reverses an incoming ball");
    check(held.diagnostics.racket_episodes == 1, "one isolated impact publishes one episode");
    check(held.contacts.size() == 1 && held.contacts.front().complete, "impact episode completes after separation");
    if (!held.contacts.empty()) {
        const auto momentum_change = subtract(held.ball_velocity,{0,0,10});
        const auto error = subtract(held.contacts.front().impulse,
                                   {.057*momentum_change.x,.057*momentum_change.y,.057*momentum_change.z});
        check(length(error) < 1e-10, "world contact impulse equals isolated ball momentum change");
    }
    check(held.diagnostics.warnings == 0, "isolated impact has no MuJoCo warning");
    const auto free = impact(true, {0,0,-.4}, {0,0,10});
    check(free.racket_velocity.z > .1, "finite-mass racket recoils");
    check(std::abs(.057*free.ball_velocity.z + .32*free.racket_velocity.z - .57) < .00001,
          "isolated free racket and ball conserve total linear momentum");
    const double initial_energy = .5*.057*100;
    check(free.diagnostics.kinetic_energy <= initial_energy*1.01, "passive impact does not create kinetic energy");
    const auto eccentric = impact(true, {.10,0,-.4}, {0,0,10});
    check(std::abs(eccentric.racket_spin.y) > .05, "off-centre hit rotates free racket");
    const auto spin = impact(false, {0,0,-.4}, {0,0,10}, {120,0,0});
    check(std::abs(spin.ball_velocity.y) > .1, "surface friction transfers incoming spin into tangential rebound");
    const auto miss = impact(false, {.30,0,-.4}, {0,0,10});
    check(miss.diagnostics.racket_episodes == 0 && std::abs(miss.ball_velocity.z - 10) < 1e-12,
          "miss remains a miss with unchanged velocity");
}

void convergence(double dt) {
    double max_velocity_ratio=0, max_spin_ratio=0, max_direction=0, max_impulse_ratio=0, max_penetration=0;
    int cases=0;
    for (bool free : {false,true}) for (const double phase : {0., .5}) {
        for (double speed : {5.,10.,25.}) for (double spin : {-300.,0.,300.}) {
        for (const double x : {0., .1, .19, .30}) {
            const auto coarse = impact(free, {x,0,-.4-speed*dt*phase}, {0,0,speed}, {spin,0,0}, dt);
            const auto fine = impact(free, {x,0,-.4-speed*dt*phase}, {0,0,speed}, {spin,0,0}, dt/2);
            const double velocity_error = length(subtract(coarse.ball_velocity, fine.ball_velocity));
            const double spin_error = length(subtract(coarse.ball_spin, fine.ball_spin));
            const double velocity_limit = std::max(.01, .01*length(fine.ball_velocity));
            const double spin_limit = std::max(1., .01*length(fine.ball_spin));
            max_velocity_ratio=std::max(max_velocity_ratio,velocity_error/velocity_limit);
            max_spin_ratio=std::max(max_spin_ratio,spin_error/spin_limit);
            max_penetration=std::max(max_penetration,coarse.diagnostics.max_penetration);
            ++cases;
            check(coarse.diagnostics.racket_episodes == fine.diagnostics.racket_episodes, "step halving retains episode count");
            check(coarse.diagnostics.racket_episodes == (x < .21 ? 1 : 0), "independent geometry predicts hit/miss episode count");
            check(velocity_error <= velocity_limit, "velocity converges within declared A3 target");
            check(spin_error <= spin_limit, "spin converges within declared A3 target");
            const double dot=coarse.ball_velocity.x*fine.ball_velocity.x+coarse.ball_velocity.y*fine.ball_velocity.y+coarse.ball_velocity.z*fine.ball_velocity.z;
            const double angle=std::acos(std::clamp(dot/(length(coarse.ball_velocity)*length(fine.ball_velocity)),-1.,1.))*180/3.14159265358979323846;
            max_direction=std::max(max_direction,angle);
            if (length(fine.ball_velocity)>1) check(angle<=.5, "rebound direction converges within half a degree");
            if (!coarse.contacts.empty() && !fine.contacts.empty()) {
                const auto& a=coarse.contacts.front();
                const auto& b=fine.contacts.front();
                const double impulse_error=length(subtract(a.impulse,b.impulse));
                const double impulse_limit=std::max(.001,.01*length(b.impulse));
                max_impulse_ratio=std::max(max_impulse_ratio,impulse_error/impulse_limit);
                check(impulse_error<=impulse_limit,"contact impulse converges within declared A3 target");
                check(std::abs(a.first_tick*dt-b.first_tick*dt/2)<=dt+1e-12,"contact onset differs by at most one coarse tick");
                check(a.complete && b.complete,"isolated episode completes at both timesteps");
            }
            check(coarse.diagnostics.warnings==0 && fine.diagnostics.warnings==0,"convergence sweep emits no solver warning");
        }
        }
    }
    std::cout << "ISOLATED_SWEEP cases="<<cases<<" dt="<<dt<<" reference_dt="<<dt/2
              <<" max_velocity_target_ratio="<<max_velocity_ratio<<" max_spin_target_ratio="<<max_spin_ratio
              <<" max_direction_degrees="<<max_direction<<" max_impulse_target_ratio="<<max_impulse_ratio
              <<" max_penetration_m="<<max_penetration<<'\n';
}

void air_and_court() {
    std::array<tennis::Snapshot,3> flight;
    std::string air=collision_fixture(false);
    replace(air,"gravity='0 0 0'","gravity='0 0 0' density='1.225' viscosity='.0000181'");
    replace(air,"size='.033'","size='.033' fluidshape='ellipsoid' fluidcoef='.5 .25 1 1 1'");
    for (std::size_t i=0;i<3;++i) {
        tennis::Simulation sim(air);
        sim.feed_ball({3,3,0},{0,0,10},{(static_cast<double>(i)-1)*100,0,0});
        sim.step(800);
        flight[i]=sim.snapshot();
    }
    check(flight[1].ball_velocity.z<10 && flight[1].ball_velocity.z>0,"air drag decelerates zero-spin flight");
    check(std::abs(flight[1].ball_velocity.y)<1e-10,"zero spin has no invented transverse lift");
    check(flight[0].ball_velocity.y*flight[2].ball_velocity.y<0,"reversed spin reverses aerodynamic lift");
    check(std::abs(flight[0].ball_velocity.y+flight[2].ball_velocity.y)<1e-10,"spherical reversed-spin flight is symmetric");
    check(std::abs(flight[0].ball_spin.x)<100 && std::abs(flight[2].ball_spin.x)<100,"air produces spin decay");

    for (double spin : {-100.,0.,100.}) {
        std::array<tennis::Snapshot,2> results;
        for (int fine=0;fine<2;++fine) {
            const double dt=runtime_timestep/(fine+1);
            std::ostringstream model;
            model<<"<mujoco><option timestep='"<<dt<<"' gravity='0 -9.81 0' integrator='implicitfast' cone='elliptic'><flag refsafe='disable'/></option>"
                 <<"<worldbody><geom name='court' type='plane' size='10 10 .1' quat='.7071067812 -.7071067812 0 0'/>"
                 <<"<body name='ball'><freejoint name='ball_free'/><inertial pos='0 0 0' mass='.057' diaginertia='.000041382 .000041382 .000041382'/>"
                 <<"<geom name='ball_geom' type='sphere' size='.033'/></body></worldbody><contact><pair geom1='ball_geom' geom2='court' condim='3' friction='.6 .6 .001 .0001 .0001' solref='-80000 -70' solimp='.95 .95 .001'/></contact></mujoco>";
            tennis::Simulation sim(model.str());sim.feed_ball({0,.4,0},{0,-5,7},{spin,0,0});
            sim.step(static_cast<std::uint64_t>(std::llround(.15/dt)));results[fine]=sim.snapshot();
        }
        check(results[0].ball_velocity.y>0,"court contact physically rebounds incoming ball");
        check(results[0].diagnostics.court_episodes==1,"one bounce emits one court episode");
        check(length(subtract(results[0].ball_velocity,results[1].ball_velocity))<=.01*length(results[1].ball_velocity),"spinning court rebound converges");
        check(std::abs(results[0].ball_spin.x-spin)>.1,"court friction changes ball spin");
    }
}

void physics_reference_checks(const std::string& production) {
    for (const bool published : {false,true}) {
        const auto profile = physics_references::measure_drag(production,published);
        const auto& samples=profile.samples;
        check(samples.size()==12,"drag fixture covers four speeds and all three axes");
        for (const auto& sample : samples) {
            check(std::abs(sample.engineering_cd-.507)<=.005,
                  std::string(published ? "published" : "production")+" ball implements engineering CD=.507 within numerical target");
            check(sample.transverse_speed<1e-10,"zero-spin spherical drag has no transverse component");
        }
        std::cout<<"DRAG_REFERENCE profile="<<(published ? "published" : "production")
                 <<" CD_15mps="<<samples.front().engineering_cd<<" CD_30mps="<<samples.back().engineering_cd<<'\n';
    }
    const auto spin=physics_references::spin_probe(production);
    check(std::abs(spin[1].acceleration.y)<1e-10,"production zero-spin ball has no transverse lift");
    check(spin[0].acceleration.y*spin[2].acceleration.y<0
          && std::abs(spin[0].acceleration.y+spin[2].acceleration.y)<1e-10,
          "production reversed spin gives symmetric opposite lift");
    check(std::abs(spin[0].outgoing_spin.x)<100 && std::abs(spin[2].outgoing_spin.x)<100,
          "production aerodynamic torque decays spin");
    const auto coarse = physics_references::emery_bounce(production,runtime_timestep);
    const auto fine = physics_references::emery_bounce(production,runtime_timestep/2);
    check(coarse.warnings==0 && fine.warnings==0,"emery comparison has no solver warning");
    check(coarse.contact.complete && fine.contact.complete,"emery comparison records a completed contact");
    check(std::abs(coarse.contact.incoming_velocity.x-2.1)<.04
          && std::abs(coarse.contact.incoming_velocity.y+.77)<.014,
          "emery contact begins within the published incident-velocity intervals");
    check(length(subtract(coarse.contact.outgoing_velocity,fine.contact.outgoing_velocity))<.01,
          "emery rebound is numerically converged below the smallest measured velocity uncertainty");
    check(length(subtract(coarse.contact.outgoing_spin,fine.contact.outgoing_spin))<.5,
          "emery spin is numerically converged within the reported spin uncertainty");
    const auto& c = coarse.contact;
    const auto change = subtract(c.outgoing_velocity,c.incoming_velocity);
    const tennis::Vec3 balance{.057*(change.x-coarse.gravity.x*c.duration),
                               .057*(change.y-coarse.gravity.y*c.duration),
                               .057*(change.z-coarse.gravity.z*c.duration)};
    check(length(subtract(balance,c.impulse))<1e-10,
          "emery solver impulse equals momentum change after accounting for inclined gravity");
    // Agreement with measured rebound is deliberately reported separately: the
    // current contact law is uncalibrated, and must not be tuned merely to pass.
    std::cout<<"EMERY_COMPARISON vx2="<<c.outgoing_velocity.x<<" published_vx2=1.23+/-0.03"
             <<" vy2="<<c.outgoing_velocity.y<<" published_vy2=0.61+/-0.03"
             <<" spin="<<-c.outgoing_spin.z<<" published_spin=52.4+/-0.5"
             <<" duration_s="<<c.duration<<'\n';
}

void control_and_replay(const std::string& model) {
    tennis::Simulation sim(model);
    check(sim.set_intent({0, {.45,.15,.3,.5,-.4,.3}, 1}), "fresh finite intent accepted");
    check(!sim.set_intent({-.5, tennis::JointValues(tennis::joint_count,0), 1}), "stale input rejected");
    check(!sim.set_intent({1, tennis::JointValues(tennis::joint_count,0), 1}), "future input rejected instead of applied early");
    check(!sim.set_intent({0, tennis::JointValues(tennis::joint_count,0), 0}), "low-confidence observation rejected");
    sim.step(100);
    const auto before_restore=sim.snapshot();
    const auto checkpoint = sim.capture();
    sim.step(200);
    const auto expected = sim.snapshot();
    sim.restore(checkpoint);
    const auto restored=sim.snapshot();
    check(std::abs(restored.diagnostics.kinetic_energy-before_restore.diagnostics.kinetic_energy)<1e-12,
          "restored energy uses restored configuration inertia, not a future mass matrix");
    sim.step(200);
    const auto replayed = sim.snapshot();
    check(length(subtract(body(expected,"ball").position, body(replayed,"ball").position)) < 1e-12,
          "checkpoint restores full integration and controller state");
    check(expected.joint_angles == replayed.joint_angles, "controller replay is exactly repeatable in the same build");
    check(expected.controller_targets==replayed.controller_targets
          && expected.diagnostics.rate_limited_ticks==replayed.diagnostics.rate_limited_ticks,
          "checkpoint preserves six-axis target slew and diagnostics mid-transition");
    check(expected.diagnostics.racket_episodes == replayed.diagnostics.racket_episodes, "replay retains event state");
    const auto paused = sim.snapshot();
    sim.step(0);
    check(sim.snapshot().time == paused.time, "zero steps implement pause without advancing time");
    sim.step(3000);
    check(sim.snapshot().diagnostics.input_expired, "missing observations expire on the simulation clock");
    for (int i=0; i<20; ++i) {
        sim.reset();
        const auto clean = sim.snapshot();
        check(clean.time == 0 && clean.tick == 0 && clean.contacts.empty(), "reset clears time, contact history and controller inputs");
    }
    sim.set_intent({0, {1e6,1e6,1e6,0,0,0}, 1});
    sim.step(20);
    const auto bounded = sim.snapshot();
    check(std::abs(bounded.actuator_efforts[0]) <= 65 && std::abs(bounded.actuator_efforts[1]) <= 65
          && std::abs(bounded.actuator_efforts[2]) <= 40, "actuation is bounded under extreme intent");

    tennis::Simulation batched(model), individual(model);
    for (int batch=0; batch<40; ++batch) {
        const auto time = batched.snapshot().time;
        const tennis::Intent intent{time,{.15*std::sin(time*12),.1,.15,.2*std::sin(time*5),-.1,.05},1};
        batched.set_intent(intent);
        individual.set_intent(intent);
        batched.step(100);
        for (int tick=0; tick<100; ++tick) {
            individual.step(1);
            static_cast<void>(individual.snapshot());
        }
    }
    const auto a = batched.snapshot();
    const auto b = individual.snapshot();
    check(a.joint_angles == b.joint_angles, "render/step batching does not alter supported controller dynamics");
    check(a.controller_targets==b.controller_targets && a.diagnostics.rate_limited_ticks==b.diagnostics.rate_limited_ticks,
          "setpoint slew is physical-tick based, not render-frame based");
    check(length(subtract(a.ball_velocity,b.ball_velocity)) == 0, "render/step batching does not alter ball trajectory");
    check(a.diagnostics.racket_episodes == b.diagnostics.racket_episodes, "batching preserves contact events");

    tennis::Simulation unstable(collision_fixture(false));
    bool stopped=false;
    try {
        unstable.feed_ball({0,1,0},{0,0,1e12});
        unstable.step(1);
    } catch (const std::exception&) { stopped=true; }
    check(stopped,"invalid finite-scale numerical input cannot silently reset the simulation on tick one");
    stopped=false;
    try { unstable.step(1); } catch (const std::exception&) { stopped=true; }
    check(stopped && unstable.snapshot().diagnostics.faulted,"faulted simulation cannot resume mixed pre/post-reset state");
    unstable.reset();
    unstable.step(1);
    check(!unstable.snapshot().diagnostics.faulted,"explicit reset clears the fault and permits a clean start");
}

tennis::Vec3 rotate_x(tennis::Vec3 v, double a) {
    return {v.x,std::cos(a)*v.y-std::sin(a)*v.z,std::sin(a)*v.y+std::cos(a)*v.z};
}
tennis::Vec3 rotate_y(tennis::Vec3 v, double a) {
    return {std::cos(a)*v.x+std::sin(a)*v.z,v.y,-std::sin(a)*v.x+std::cos(a)*v.z};
}
tennis::Vec3 rotate_z(tennis::Vec3 v, double a) {
    return {std::cos(a)*v.x-std::sin(a)*v.y,std::sin(a)*v.x+std::cos(a)*v.y,v.z};
}
tennis::Vec3 upper_rotation(tennis::Vec3 v, const tennis::JointValues& q) {
    return rotate_y(rotate_z(rotate_x(v,q[3]),q[1]),q[0]);
}

// Public behavior of the six-coordinate arm, independent of MuJoCo qpos order.
void six_axis_control(const std::string& model) {
    tennis::Simulation sim(model);
    const auto initial=sim.snapshot();
    check(initial.joint_angles.size()==6,"six physical joint coordinates are exposed");
    check(length(subtract(body(initial,"hand").position,body(initial,"racket").position))<1e-12,
          "finite hand and rigid grip share the declared origin without adding reach");
    const tennis::JointValues rates{3,3,4,4,5,4}, caps{65,65,40,20,8,6};
    const tennis::JointValues command{1,-1,1,.8,.7,-.5};
    sim.set_intent({0,command,1});
    auto previous=sim.snapshot();
    check(previous.joint_angles==initial.joint_angles && previous.controller_targets==initial.controller_targets,
          "accepting an input cannot teleport physical or controller state");
    for (int tick=0;tick<2100;++tick) {
        sim.step(1);const auto now=sim.snapshot();
        for (std::size_t i=0;i<tennis::joint_count;++i) {
            check(std::abs(now.controller_targets[i]-previous.controller_targets[i])<=rates[i]*now.timestep+1e-12,
                  "all six controller targets obey rate bounds, including input expiry");
            check(std::abs(now.actuator_efforts[i])<=caps[i]+1e-12,"every joint effort remains bounded");
        }
        previous=now;
    }
    check(previous.diagnostics.input_expired && previous.diagnostics.rate_limited_ticks>0,
          "dropout expires and slews back instead of retaining a stale six-axis request");
    check(previous.diagnostics.target_clipped_ticks==0,"valid in-envelope targets are not range clipping");
    const auto before_feed=sim.snapshot();
    sim.feed_ball({3,3,0},{0,0,0});
    check(sim.snapshot().controller_targets==before_feed.controller_targets,"feed does not erase controller history");
    sim.reset();
    check(sim.snapshot().controller_targets==initial.controller_targets
          && sim.snapshot().diagnostics.rate_limited_ticks==0,"reset restores target history and counters");
    sim.set_intent({0,{1e6,1e6,1e6,1e6,1e6,1e6},1});sim.step(1);
    const auto clipped=sim.snapshot();
    check(clipped.diagnostics.target_clipped_ticks==1 && clipped.diagnostics.torque_saturated_ticks==0,
          "one range-clipped tick is counted separately from effort saturation");
    auto weak=model;
    replace(weak,"ctrlrange=\"-65 65\"","ctrlrange=\"-.0001 .0001\"");
    replace(weak,"ctrlrange=\"-65 65\"","ctrlrange=\"-.0001 .0001\"");
    tennis::Simulation weak_sim(weak);weak_sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1});weak_sim.step(1);
    check(weak_sim.snapshot().diagnostics.torque_saturated_ticks==1
          && weak_sim.snapshot().diagnostics.target_clipped_ticks==0,
          "finite gravity compensation can saturate effort without clipping the target");
    auto invalid=command;invalid[5]=std::numeric_limits<double>::quiet_NaN();
    check(!sim.set_intent({sim.snapshot().time,invalid,1}),"nonfinite sixth coordinate is rejected");
    tennis::Simulation mixed(model);mixed.feed_ball({3,3,0},{0,0,0});
    for (int frame=0;frame<180;++frame) {
        mixed.set_intent({mixed.snapshot().time,{.3,-.6,1.,.8,.2,-.15},1});mixed.step(80);
    }
    const auto m=mixed.snapshot();
    const auto expected_upper=upper_rotation({.30,0,0},m.joint_angles);
    const auto expected_forearm=upper_rotation(rotate_y({.28,0,0},m.joint_angles[2]),m.joint_angles);
    check(length(subtract(subtract(body(m,"forearm").position,body(m,"upper_arm").position),expected_upper))<1e-10,
          "mixed-pose upper-arm geometry independently matches Ry(yaw) Rz(pitch) Rx(axial)");
    check(length(subtract(subtract(body(m,"hand").position,body(m,"forearm").position),expected_forearm))<1e-10,
          "mixed-pose forearm direction preserves elbow plane and public elbow index");
    const auto expected_hand_x=upper_rotation(rotate_y(rotate_x(rotate_y({1,0,0},m.joint_angles[5]),m.joint_angles[4]),m.joint_angles[2]),m.joint_angles);
    const auto h=body(m,"hand").orientation;
    const tennis::Vec3 actual_hand_x{1-2*(h.y*h.y+h.z*h.z),2*(h.x*h.y+h.w*h.z),2*(h.x*h.z-h.w*h.y)};
    check(length(subtract(actual_hand_x,expected_hand_x))<1e-10,
          "mixed-pose hand rotation follows forearm roll then wrist bend");
    check(length(subtract(body(m,"hand").position,body(m,"racket").position))<1e-12,
          "wrist bend cannot translate the rigid grip origin");
    // A forearm-only roll should rotate the face while preserving limb directions
    // after a slow settle; it must be an actual controlled physical coordinate.
    tennis::Simulation rolling(model);rolling.feed_ball({3,3,0},{0,0,0});
    for (int frame=0;frame<180;++frame) {
        rolling.set_intent({rolling.snapshot().time,{0,0,0,0,.3,0},1});rolling.step(80);
    }
    const auto rolled=rolling.snapshot();
    check(std::abs(rolled.joint_angles[4]-.3)<.015,"forearm roll tracks a physical finite-effort target");
    check(length(subtract(body(rolled,"hand").position,body(initial,"hand").position))<.005,
          "forearm axial roll preserves wrist location in the isolated neutral pose");
    check(std::abs(body(rolled,"racket").orientation.x-body(initial,"racket").orientation.x)>.05,
          "distal physical roll changes racket face orientation");
    check(rolled.diagnostics.racket_episodes==0,"face control cannot invent a contact with a distant ball");
}

void production_drill(const std::string& model) {
    std::array<tennis::Snapshot,2> variations;
    for (int variant=0;variant<2;++variant) {
        std::array<tennis::Snapshot,2> outputs;
        for (int fine=0;fine<2;++fine) {
            auto xml=model;
            if (fine) replace(xml,"timestep=\"0.000125\"","timestep=\"0.0000625\"");
            tennis::Simulation sim(xml);
            const double dt=sim.snapshot().timestep;
            for (int frame=0;frame<60;++frame) {
                const tennis::JointValues target = variant ? tennis::JointValues{.04,.03,.08,.10,.08,-.04} : tennis::JointValues(tennis::joint_count,0);
                sim.set_intent({frame*.01,target,1});
                sim.step(static_cast<std::uint64_t>(std::llround(.01/dt)));
            }
            outputs[fine]=sim.snapshot();
        }
        const auto& a=outputs[0];const auto& b=outputs[1];
        check(a.diagnostics.racket_episodes==1 && b.diagnostics.racket_episodes==1,"production arm returns the default feed through actual contact");
        check(a.ball_velocity.z<0,"production return travels toward the net");
        check(a.diagnostics.max_racket_speed>.05 && a.diagnostics.max_stringbed_speed>=a.diagnostics.max_racket_speed,
              "coupled production racket recoils and reports COM/head speed separately");
        check(length(subtract(a.ball_velocity,b.ball_velocity))<=std::max(.01,.01*length(b.ball_velocity)),"arm-attached return velocity converges");
        check(length(subtract(a.ball_spin,b.ball_spin))<=std::max(1.,.01*length(b.ball_spin)),"arm-attached spin converges");
        if (!a.contacts.empty() && !b.contacts.empty()) {
            check(length(subtract(a.contacts[0].impulse,b.contacts[0].impulse))<=std::max(.001,.01*length(b.contacts[0].impulse)),
                  "arm-attached integrated impulse converges");
            const auto av=a.contacts[0].outgoing_velocity, bv=b.contacts[0].outgoing_velocity;
            const double direction_error=std::acos(std::clamp((av.x*bv.x+av.y*bv.y+av.z*bv.z)/(length(av)*length(bv)),-1.,1.))*180/std::acos(-1.);
            const double onset_error=std::abs(a.contacts[0].first_tick*a.timestep-b.contacts[0].first_tick*b.timestep);
            check(direction_error<=.5,"six-axis arm return direction converges within half a degree");
            check(onset_error<=runtime_timestep+1e-12,"six-axis contact onset converges within one runtime tick");
            std::cout<<"ARM_CONVERGENCE variant="<<variant<<" direction_error_deg="<<direction_error
                     <<" onset_error_s="<<onset_error<<" velocity_error_mps="<<length(subtract(av,bv))
                     <<" spin_error_rad_s="<<length(subtract(a.contacts[0].outgoing_spin,b.contacts[0].outgoing_spin))
                     <<" impulse_error_Ns="<<length(subtract(a.contacts[0].impulse,b.contacts[0].impulse))<<'\n';
            std::cout<<"PRODUCTION_DRILL variant="<<variant<<" incoming_mps="<<length(a.contacts[0].incoming_velocity)
                     <<" outgoing_mps="<<length(a.contacts[0].outgoing_velocity)<<" outgoing_spin="<<length(a.contacts[0].outgoing_spin)
                     <<" duration_s="<<a.contacts[0].duration<<" penetration_m="<<a.contacts[0].max_penetration
                     <<" max_racket_COM_mps="<<a.diagnostics.max_racket_speed<<" max_stringbed_mps="<<a.diagnostics.max_stringbed_speed
                     <<" actuator_work_J="<<a.diagnostics.actuator_work<<'\n';
        }
        check(a.diagnostics.warnings==0 && b.diagnostics.warnings==0,"production motion/contact produces no solver warning");
        variations[variant]=a;
    }
    check(length(subtract(variations[0].ball_velocity,variations[1].ball_velocity))>.5,
          "physical racket motion/orientation changes return velocity");
    check(length(subtract(variations[0].ball_spin,variations[1].ball_spin))>1,
          "physical racket motion/orientation changes return spin");

    tennis::Simulation drill(model);
    bool crossed=false, landed=false;
    for (int frame=0;frame<200;++frame) {
        drill.set_intent({frame*.01,{0,0,0,0,0,0},1});
        drill.step(80);
        const auto s=drill.snapshot();
        const auto position=body(s,"ball").position;
        if (position.z<0 && position.y>.914+.033 && !crossed) crossed=true;
        for (const auto& event:s.contacts) if (event.surface=="court" && event.position.z<0) landed=true;
    }
    check(crossed && landed,"neutral supported drill clears regulation net height then physically lands across court");

    // Same feed, different bounded physical intentions. Neither preset edits the
    // ball after setup. This qualifies two stationary returns, not stroke families.
    std::array<tennis::ContactEvent,2> impacts;
    std::array<tennis::Vec3,2> faces;
    const tennis::JointValues nominal_min{-1.3,-1.4,0,-1.2,-std::acos(-1.)/4,-std::acos(-1.)/6};
    const tennis::JointValues nominal_max{1.3,.8,2.269,1.2,std::acos(-1.)/4,std::acos(-1.)/6};
    // Declared before this per-tick rerun after review measured about .010 rad
    // elbow soft-stop penetration. This is an engineering regression guard for
    // these demonstrated paths, not anatomical validation or a hard qpos clamp.
    constexpr double soft_stop_allowance = .02;
    for (int variant=0;variant<2;++variant) {
        tennis::Simulation returning(model);
        const tennis::JointValues target = variant ? tennis::JointValues{.04,.03,.08,.10,.08,-.04} : tennis::JointValues(tennis::joint_count,0);
        bool net_clear=false,landed=false,face_recorded=false;
        tennis::Vec3 landing{};
        auto observed_min=returning.snapshot().joint_angles, observed_max=observed_min;
        double maximum_overshoot=0;
        for (int frame=0;frame<350;++frame) {
            returning.set_intent({returning.snapshot().time,target,1});
            for (int tick=0;tick<80;++tick) {
                returning.step(1);
                const auto physical=returning.snapshot();
                for (std::size_t i=0;i<tennis::joint_count;++i) {
                    const double angle=physical.joint_angles[i];
                    observed_min[i]=std::min(observed_min[i],angle);
                    observed_max[i]=std::max(observed_max[i],angle);
                    maximum_overshoot=std::max({maximum_overshoot,nominal_min[i]-angle,angle-nominal_max[i]});
                }
            }
            const auto s=returning.snapshot();const auto p=body(s,"ball").position;
            if (p.z<0 && p.y>.947) net_clear=true;
            for (const auto& event:s.contacts) {
                if (event.surface=="racket" && event.complete) impacts[variant]=event;
                if (event.surface=="court" && !landed) {landed=true;landing=event.position;}
            }
            if (s.diagnostics.racket_episodes && !face_recorded) {
                const auto q=body(s,"racket").orientation;
                faces[variant]={2*(q.x*q.z+q.w*q.y),2*(q.y*q.z-q.w*q.x),1-2*(q.x*q.x+q.y*q.y)};
                face_recorded=true;
            }
        }
        const auto s=returning.snapshot();
        check(s.diagnostics.racket_episodes==1 && s.diagnostics.net_episodes==0 && net_clear && landed,
              "both same-feed reference intentions physically return over net without a net contact");
        check(landing.z<0 && landing.z>=-11.885 && std::abs(landing.x)<=4.115,
              "each qualified reference has a first landing inside the opposite singles court");
        check(s.diagnostics.warnings==0 && s.diagnostics.target_clipped_ticks==0,
              "qualified reference intentions need no range clipping and produce no solver warning");
        check(maximum_overshoot<=soft_stop_allowance,
              "actual reference joint motion stays within the declared .02 rad soft-stop regression allowance");
        std::cout<<"JOINT_EXTREMA variant="<<variant<<" max_overshoot_rad="<<maximum_overshoot<<" minimum=[";
        for (std::size_t i=0;i<tennis::joint_count;++i) {if(i) std::cout<<',';std::cout<<observed_min[i];}
        std::cout<<"] maximum=[";
        for (std::size_t i=0;i<tennis::joint_count;++i) {if(i) std::cout<<',';std::cout<<observed_max[i];}
        std::cout<<"] allowance_rad="<<soft_stop_allowance<<" sampled_every_physics_tick=true\n";
        std::cout<<"SIX_AXIS_RETURN variant="<<variant<<" first_landing_xz="<<landing.x<<','<<landing.z
                 <<" velocity="<<impacts[variant].outgoing_velocity.x<<','<<impacts[variant].outgoing_velocity.y<<','<<impacts[variant].outgoing_velocity.z
                 <<" spin="<<impacts[variant].outgoing_spin.x<<','<<impacts[variant].outgoing_spin.y<<','<<impacts[variant].outgoing_spin.z<<'\n';
    }
    check(length(subtract(faces[0],faces[1]))>.05,"preset changes produce a measured racket-face change");
    check(length(subtract(impacts[0].outgoing_velocity,impacts[1].outgoing_velocity))>1
          && length(subtract(impacts[0].outgoing_spin,impacts[1].outgoing_spin))>1,
          "changed physical path/face causally changes outgoing ball velocity and spin");
    // Hold all shoulder/elbow intentions identical to the neutral reference;
    // change only distal controls to isolate their effect on ball response.
    tennis::Simulation distal(model);
    for (int frame=0;frame<60;++frame) {
        distal.set_intent({distal.snapshot().time,{0,0,0,0,.10,.04},1});distal.step(80);
    }
    const auto distal_result=distal.snapshot();
    check(distal_result.diagnostics.racket_episodes==1 && !distal_result.contacts.empty()
          && distal_result.contacts[0].complete,"distal-only comparison contains a real completed hit");
    if (!distal_result.contacts.empty()) {
        const auto& d=distal_result.contacts[0];
        const double velocity_change=length(subtract(d.outgoing_velocity,impacts[0].outgoing_velocity));
        const double spin_change=length(subtract(d.outgoing_spin,impacts[0].outgoing_spin));
        check(velocity_change>1 && spin_change>1,
              "changing only distal intentions changes physical outgoing velocity/spin beyond numerical errors");
        std::cout<<"DISTAL_CAUSAL velocity_change_mps="<<velocity_change<<" spin_change_rad_s="<<spin_change<<'\n';
    }
    tennis::Simulation miss(model);
    for (int frame=0;frame<100;++frame) {
        miss.set_intent({miss.snapshot().time,{.2,.25,.8,.8,0,0},1});miss.step(80);
    }
    check(miss.snapshot().diagnostics.racket_episodes==0 && miss.snapshot().ball_velocity.z>0,
          "an unsuitable bounded arm path misses the same feed instead of receiving a repaired hit");

    tennis::Simulation active_arm(model);
    bool arm_impact_found=false;
    for (int tick=0;tick<4000;++tick) {
        const auto pre=active_arm.snapshot();
        active_arm.set_intent({pre.time,{.04,.03,.08,.10,.08,-.04},1});active_arm.step(1);
        const auto during_arm=active_arm.snapshot();
        if (during_arm.contacts.empty() || during_arm.contacts[0].surface!="racket" || during_arm.contacts[0].complete) continue;
        const auto saved=active_arm.capture();active_arm.step(400);const auto expected=active_arm.snapshot();
        active_arm.restore(saved);active_arm.step(400);const auto repeated=active_arm.snapshot();
        check(expected.joint_angles==repeated.joint_angles && expected.controller_targets==repeated.controller_targets,
              "six-axis active-impact replay restores physical and target-history state exactly");
        check(length(subtract(expected.contacts[0].impulse,repeated.contacts[0].impulse))==0,
              "six-axis active-impact replay preserves accumulated impulse");
        arm_impact_found=true;break;
    }
    check(arm_impact_found,"six-axis replay fixture actually checkpoints an active contact");

    tennis::Simulation active(collision_fixture(true));
    active.feed_ball({0,0,-.4},{0,0,10},{100,0,0});
    active.step(300); // .0375s: inside the contact episode, rather than before it.
    const auto during=active.snapshot();
    const auto checkpoint=active.capture();
    active.step(400);
    const auto complete=active.snapshot();
    active.restore(checkpoint);active.step(400);const auto repeated=active.snapshot();
    check(!during.contacts.empty() && !during.contacts[0].complete,"checkpoint fixture captures an active impact");
    check(complete.contacts.size()==1 && repeated.contacts.size()==1,"active-impact replay does not duplicate episodes");
    if (!complete.contacts.empty() && !repeated.contacts.empty()) {
        check(length(subtract(complete.contacts[0].impulse,repeated.contacts[0].impulse))==0,
              "active-impact checkpoint restores accumulated contact impulse exactly");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string models = argc > 1 ? argv[1] : "models";
        const auto production=read(models + "/supported_arm.xml");
        if (argc>2 && std::string(argv[2])=="--physics-report") {
            physics_references::write_report(std::cout,production);
            return 0;
        }
        free_flight();
        contact_behaviors();
        convergence(argc>2 && std::string(argv[2])=="--coarse-regression" ? .0005 : runtime_timestep);
        air_and_court();
        physics_reference_checks(production);
        control_and_replay(production);
        six_axis_control(production);
        production_drill(production);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << ": " << failures << " behavioral checks failed\n";
    return failures == 0 ? 0 : 1;
}
