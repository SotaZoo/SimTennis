#pragma once

#include "simulation.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace physics_references {

// These are measurement comparators, not an alternative physics implementation.
struct DragSample {
    int axis;
    double speed;
    double force;
    double engineering_cd;
    double transverse_speed;
};

// Both profiles inherit every fluid coefficient from the production XML.
// The published profile substitutes only ball mass/radius/inertia and air density.
struct DragProfile {
    std::string configuration;
    std::vector<DragSample> samples;
};
DragProfile measure_drag(const std::string& production_xml, bool published_profile);

struct SpinSample {
    double incoming_spin;
    tennis::Vec3 acceleration;
    tennis::Vec3 outgoing_spin;
};
// Production fluid settings; gravity removed; velocity +Z 20 m/s; spin about X.
// The returned acceleration is one-tick delta-v/dt, not an empirical lift fit.
std::vector<SpinSample> spin_probe(const std::string& production_xml);

struct EmeryBounce {
    std::string configuration;
    double timestep;
    tennis::Vec3 gravity;
    tennis::ContactEvent contact;
    std::uint64_t warnings;
};

// Cross (2002), Table I: zero-spin 57 g ball, 20 degree incidence on emery.
// Keeps the production normal contact law, but uses the paper's emery friction
// and approximate finite-wall inertia. Gravity is in the surface coordinate frame.
EmeryBounce emery_bounce(const std::string& production_xml, double timestep);

// JSON is evidence: a completed report can explicitly have a failed empirical
// comparison. Its status must not be confused with the numerical test exit code.
void write_report(std::ostream& out, const std::string& production_xml);

} // namespace physics_references
