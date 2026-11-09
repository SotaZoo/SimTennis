#include "simulation.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool good, const char* reason) {
    if (!good) throw std::runtime_error(reason);
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Expected car model path");
        std::ifstream file(argv[1]);
        const std::string xml{std::istreambuf_iterator<char>(file), {}};
        tennis::Simulation sim(xml);
        const auto initial = sim.snapshot();
        check(sim.set_effort({0, {40, 0, 0}, {350, 0, -350}}), "Direct effort accepts");
        sim.step(160);
        auto state = sim.snapshot();
        check(state.players[0].base_velocity.x > .1 && state.players[0].base_velocity.z < -.1,
              "Both planar axes accelerate from simultaneous force");
        check(state.players[0].joint_velocities[0] > .1, "Torque physically accelerates swing");
        check(state.players[0].actuator_efforts == tennis::JointValues({40, 0, 0}),
              "Direct torques bypass position servos and bias compensation");
        check(state.players[0].controller_targets == initial.players[0].controller_targets,
              "Direct effort does not select a target trajectory");
        check(std::abs(state.players[1].base_velocity.x) < 1e-8,
              "Other player's commands are independent");

        check(!sim.set_effort({state.time, {NAN, 0, 0}, {}}), "Nonfinite torque rejects");
        check(!sim.set_effort({state.time, {0}, {}}), "Wrong-sized torque rejects");
        check(!sim.set_effort({state.time, {0, 0, 0}, {0, 1, 0}}), "Vertical force rejects");
        check(!sim.set_effort({state.time + 1, {0, 0, 0}, {}}), "Future command rejects");
        check(!sim.set_effort({state.time, {0, 0, 0}, {}}, 2), "Missing player rejects");
        check(sim.set_effort({state.time, {100, 100, -100}, {1000, 0, 1000}}), "Finite large effort clamps");
        sim.step(1);
        state = sim.snapshot();
        check(state.players[0].actuator_efforts == tennis::JointValues({40, 6, -40}), "Model torque caps apply");
        check(std::abs(std::hypot(state.players[0].movement_effort.x, state.players[0].movement_effort.z) - 700) < 1e-8,
              "Planar resultant force is bounded");

        const auto checkpoint = sim.capture();
        sim.step(160);
        const auto continued = sim.snapshot();
        sim.restore(checkpoint);
        sim.step(80); sim.step(80);
        const auto replayed = sim.snapshot();
        check(continued.players[0].joint_angles == replayed.players[0].joint_angles,
              "Checkpoint includes effort mode and values");
        check(continued.players[0].base_position.x == replayed.players[0].base_position.x,
              "Effort replay is independent of tick chunking");

        sim.reset();
        check(sim.set_effort({0, {0, 0, 0}, {100, 0, 0}}), "Apply support force");
        sim.step(2400);
        const double speed = sim.snapshot().players[0].base_velocity.x;
        check(sim.set_effort({sim.snapshot().time, {0, 0, 0}, {}}), "Release to zero effort");
        sim.step(800);
        state = sim.snapshot();
        check(std::abs(state.players[0].base_velocity.x - speed) < .02,
              "Zero effort does not invoke hidden velocity braking");
        sim.step(2400);
        state = sim.snapshot();
        check(state.players[0].input_expired && state.players[0].movement_effort.x == 0,
              "Expired direct effort becomes zero force");
        check(sim.set_intent({state.time, {1.57, 0, 0}, 1, {}}), "Servo input remains available");
        sim.step(160);
        check(sim.snapshot().players[0].movement_effort.x < 0, "Explicit servo input restores its braking controller");
        check(sim.snapshot().diagnostics.warnings == 0, "No numerical warnings");
        std::cout << "PASS: bounded direct effort, movement, rejection, expiration and replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
