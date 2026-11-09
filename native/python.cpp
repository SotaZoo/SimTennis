#include "simulation.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
namespace {
std::array<double, 3> values(tennis::Vec3 v) { return {v.x, v.y, v.z}; }
tennis::Vec3 vec(std::array<double, 3> v) { return {v[0], v[1], v[2]}; }
py::dict bodies(const std::vector<tennis::BodyPose>& poses) {
    py::dict result;
    for (const auto& pose : poses)
        result[py::str(pose.name)] = py::dict("position"_a=values(pose.position),
            "orientation"_a=std::array<double,4>{pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z});
    return result;
}
}

PYBIND11_MODULE(_tennis_native, module) {
    module.attr("build_id") = TENNIS_NATIVE_BUILD_ID;
    py::class_<tennis::CapturedState>(module, "CapturedState");
    py::class_<tennis::Simulation>(module, "Simulation")
        .def(py::init<const std::string&>())
        .def("reset", &tennis::Simulation::reset)
        .def("set_player_start", [](tennis::Simulation& sim, std::array<double,2> position, int player) {
            const auto state = sim.snapshot();
            if (player < 0 || static_cast<std::size_t>(player) >= state.players.size())
                throw std::invalid_argument("Player start requires an existing player");
            sim.set_player_start({position[0], state.players[player].base_position.y, position[1]}, player);
        }, "position"_a, "player"_a=0)
        .def("feed_ball", [](tennis::Simulation& sim, std::array<double,3> p,
                std::array<double,3> v, std::array<double,3> w) { sim.feed_ball(vec(p),vec(v),vec(w)); },
                "position"_a, "velocity"_a, "spin"_a=std::array<double,3>{})
        .def("set_effort", [](tennis::Simulation& sim, double time, const tennis::JointValues& torques,
                std::array<double,2> force, int player) { return sim.set_effort({time,torques,{force[0],0,force[1]}},player); },
                "time"_a, "torques"_a, "force"_a, "player"_a=0)
        .def("step", &tennis::Simulation::step, py::call_guard<py::gil_scoped_release>())
        .def("capture", &tennis::Simulation::capture)
        .def("restore", &tennis::Simulation::restore)
        .def("configuration", &tennis::Simulation::configuration)
        .def("snapshot", [](const tennis::Simulation& sim) {
            const auto s = sim.snapshot();
            py::list players, contacts;
            for (const auto& p : s.players)
                players.append(py::dict("position"_a=values(p.base_position), "velocity"_a=values(p.base_velocity),
                    "joint_angles"_a=p.joint_angles, "joint_velocities"_a=p.joint_velocities,
                    "torques"_a=p.actuator_efforts, "force"_a=values(p.movement_effort),
                    "racket_velocity"_a=values(p.racket_velocity), "bodies"_a=bodies(p.bodies)));
            for (const auto& e : s.contacts)
                contacts.append(py::dict("surface"_a=e.surface, "player"_a=e.player,
                    "first_tick"_a=e.first_tick, "last_tick"_a=e.last_tick,
                    "position"_a=values(e.position), "duration"_a=e.duration,
                    "incoming_velocity"_a=values(e.incoming_velocity),
                    "outgoing_velocity"_a=values(e.outgoing_velocity), "complete"_a=e.complete));
            return py::dict("tick"_a=s.tick, "time"_a=s.time, "timestep"_a=s.timestep,
                "bodies"_a=bodies(s.bodies), "ball_velocity"_a=values(s.ball_velocity),
                "ball_spin"_a=values(s.ball_spin), "players"_a=players, "contacts"_a=contacts,
                "warnings"_a=s.diagnostics.warnings, "faulted"_a=s.diagnostics.faulted,
                "actuator_work"_a=s.diagnostics.actuator_work);
        });
}
