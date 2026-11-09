#include "simulation.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
void vector(tennis::Vec3 v) { std::cout << '[' << v.x << ',' << v.y << ',' << v.z << ']'; }
void joints(const tennis::JointValues& v) {
    std::cout << '[';
    for (std::size_t i=0;i<v.size();++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}
tennis::Vec3 position(const tennis::Snapshot& s, const std::string& name) {
    for (const auto& body : s.bodies) if (body.name == name) return body.position;
    throw std::runtime_error("Missing physical body " + name);
}
}

// The same bounded input sequence runs in the packaged application's QA mode.
// JSON numbers retain double precision here; Godot's Vector3 renders at float.
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: native_trace model.xml");
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("Cannot read model");
        const std::string xml{std::istreambuf_iterator<char>(file), {}};
        tennis::Simulation simulation(xml);
        auto targets=simulation.snapshot().joint_angles;
        if (targets.size()==tennis::joint_count) targets={.1,.08,.05,.04,-.03,.02};
        else for (std::size_t i=0;i<targets.size();++i) targets[i]+=.1/static_cast<double>(i+1);
        std::cout << std::setprecision(17) << "{\"samples\":[";
        for (int i=0; i<=40; ++i) {
            const auto s = simulation.snapshot();
            if (i) std::cout << ',';
            std::cout << "{\"tick\":" << s.tick << ",\"time\":" << s.time << ",\"ball_position\":";
            vector(position(s, "ball"));
            std::cout << ",\"ball_velocity\":"; vector(s.ball_velocity);
            std::cout << ",\"ball_spin\":"; vector(s.ball_spin);
            std::cout << ",\"racket_position\":"; vector(position(s, "racket"));
            std::cout << ",\"joint_angles\":"; joints(s.joint_angles);
            std::cout << ",\"controller_targets\":"; joints(s.controller_targets);
            std::cout << '}';
            if (i < 40) {
                // PackedFloat64Array preserves these doubles at the Godot boundary.
                if (!simulation.set_intent({s.time,targets,1}))
                    throw std::runtime_error("Trace command was rejected by the model");
                simulation.step(400);
            }
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
