#include "physics_references.hpp"

#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace physics_references {
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double published_cd = .507;
constexpr double numerical_cd_tolerance = .005; // Engineering acceptance, not measurement uncertainty.
constexpr double ball_mass = .057;
constexpr double ball_radius = .033;

void replace(std::string& text, const std::string& from, const std::string& to) {
    const auto position = text.find(from);
    if (position == std::string::npos) throw std::runtime_error("Reference fixture missing token: " + from);
    text.replace(position, from.size(), to);
}

std::string number(double value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
}

std::unique_ptr<mjModel, decltype(&mj_deleteModel)> load_model(const std::string& xml) {
    mjVFS files;
    mj_defaultVFS(&files);
    if (mj_addBufferVFS(&files, "reference.xml", xml.data(), static_cast<int>(xml.size())) != 0) {
        mj_deleteVFS(&files);
        throw std::runtime_error("Cannot allocate reference model");
    }
    char error[1024]{};
    auto* model = mj_loadXML("reference.xml", &files, error, sizeof(error));
    mj_deleteVFS(&files);
    if (!model) throw std::runtime_error(error);
    return {model, mj_deleteModel};
}

double component(tennis::Vec3 v, int axis) { return axis == 0 ? v.x : axis == 1 ? v.y : v.z; }

void vector(std::ostream& out, tennis::Vec3 v) { out << '[' << v.x << ',' << v.y << ',' << v.z << ']'; }

// Table I measured/derived values and the paper's reported uncertainties. The
// latter are not asserted to be independent, Gaussian, or confidence intervals.
struct Comparison { const char* name; const char* unit; double expected; double uncertainty; double actual; };
std::array<Comparison, 6> comparisons(const EmeryBounce& bounce) {
    const auto& c = bounce.contact;
    return {{{"tangential_rebound", "m/s", 1.23, .03, c.outgoing_velocity.x},
             {"normal_rebound", "m/s", .61, .03, c.outgoing_velocity.y},
             {"clockwise_spin", "rad/s", 52.4, .5, -c.outgoing_spin.z},
             // Paper Eqs. 8-9 neglect gravity. Compare its convention separately
             // from the solver contact impulse, which excludes gravity.
             {"tangential_momentum_change", "N s", .050, .004,
              ball_mass*(c.incoming_velocity.x-c.outgoing_velocity.x)},
             {"normal_momentum_change", "N s", .079, .002,
              ball_mass*(c.outgoing_velocity.y-c.incoming_velocity.y)},
             {"spin_to_roll_ratio", "1", 1.41, .05,
              -ball_radius*c.outgoing_spin.z/c.outgoing_velocity.x}}};
}
} // namespace

DragProfile measure_drag(const std::string& production_xml, bool published_profile) {
    auto xml = production_xml;
    replace(xml, "gravity=\"0 -9.81 0\"", "gravity=\"0 0 0\"");
    if (published_profile) {
        // Cross & Lindsey's measured dimensions, not a separately copied fluidcoef.
        replace(xml, "mass=\".057\"", "mass=\".05803\"");
        replace(xml, "size=\".033\"", "size=\".03284\"");
        const auto inertia = number((2./3)*.05803*.03284*.03284);
        replace(xml, ".000041382 .000041382 .000041382", inertia+" "+inertia+" "+inertia);
        replace(xml, "density=\"1.225\"", "density=\"1.21\"");
    }
    const auto model = load_model(xml);
    const int ball = mj_name2id(model.get(), mjOBJ_BODY, "ball");
    const int geom = mj_name2id(model.get(), mjOBJ_GEOM, "ball_geom");
    if (ball < 0 || geom < 0) throw std::runtime_error("Drag reference requires the production ball");
    const double mass = model->body_mass[ball];
    const double radius = model->geom_size[3*geom];
    const double density = model->opt.density;
    DragProfile result{tennis::Simulation(xml).configuration(),{}};
    for (const double speed : {15., 20., 25., 30.}) for (int axis=0; axis<3; ++axis) {
        tennis::Simulation sim(xml);
        tennis::Vec3 velocity{};
        if (axis == 0) velocity.x = speed;
        else if (axis == 1) velocity.y = speed;
        else velocity.z = speed;
        sim.feed_ball({20,20,20}, velocity);
        sim.step(1);
        const auto state = sim.snapshot();
        const double force = mass*(speed-component(state.ball_velocity,axis))/state.timestep;
        double transverse_squared = 0;
        for (int other=0; other<3; ++other) if (other != axis)
            transverse_squared += std::pow(component(state.ball_velocity,other),2);
        result.samples.push_back({axis, speed, force, force/(.5*density*pi*radius*radius*speed*speed),
                                  std::sqrt(transverse_squared)});
    }
    return result;
}

std::vector<SpinSample> spin_probe(const std::string& production_xml) {
    auto xml = production_xml;
    replace(xml,"gravity=\"0 -9.81 0\"","gravity=\"0 0 0\"");
    std::vector<SpinSample> samples;
    for (const double spin : {-100.,0.,100.}) {
        tennis::Simulation sim(xml);
        sim.feed_ball({20,20,20},{0,0,20},{spin,0,0});
        sim.step(1);
        const auto state=sim.snapshot();
        samples.push_back({spin,{state.ball_velocity.x/state.timestep,
                                state.ball_velocity.y/state.timestep,
                                (state.ball_velocity.z-20)/state.timestep},state.ball_spin});
    }
    return samples;
}

EmeryBounce emery_bounce(const std::string& production_xml, double timestep) {
    auto xml = production_xml;
    const tennis::Vec3 gravity{9.81*std::cos(20*pi/180), -9.81*std::sin(20*pi/180), 0};
    replace(xml, "gravity=\"0 -9.81 0\"", "gravity=\""+number(gravity.x)+" "+number(gravity.y)+" 0\"");
    replace(xml, "timestep=\"0.000125\"", "timestep=\""+number(timestep)+"\"");
    replace(xml, "density=\"1.225\"", "density=\"0\"");
    replace(xml, "viscosity=\"0.0000181\"", "viscosity=\"0\"");
    replace(xml, "friction=\".6 .6 .001 .0001 .0001\"", "friction=\".67 .67 .001 .0001 .0001\"");
    // Paper Sec. IV approximates I=2/3*m*(30 mm)^2, not a measured inertia.
    replace(xml, ".000041382 .000041382 .000041382", ".0000342 .0000342 .0000342");
    tennis::Simulation sim(xml);
    // Start at geometric contact. Capture the actual first positive-force input,
    // avoiding gravitational acceleration from a long artificial approach.
    sim.feed_ball({20,ball_radius,20}, {2.10,-.77,0});
    for (int tick=0; tick<static_cast<int>(std::ceil(.05/timestep)); ++tick) {
        sim.step(1);
        const auto state = sim.snapshot();
        for (const auto& contact : state.contacts) if (contact.surface == "court" && contact.complete)
            return {sim.configuration(), timestep, gravity, contact, state.diagnostics.warnings};
    }
    throw std::runtime_error("Emery reference did not produce one completed bounce");
}

void write_report(std::ostream& out, const std::string& production_xml) {
    const auto production_drag=measure_drag(production_xml,false);
    const auto published_drag=measure_drag(production_xml,true);
    out << std::setprecision(12)
        << "{\n  \"production_configuration\": " << std::quoted(tennis::Simulation(production_xml).configuration())
        << ",\n  \"drag\": {\n    \"source\": \"https://doi.org/10.1007/s12283-013-0144-9\",\n"
        << "    \"author_methods\": \"https://twu.tennis-warehouse.com/learning_center/aerodynamics2.php\",\n"
        << "    \"published_mean_cd\": 0.507, \"published_shot_sd\": 0.024,\n"
        << "    \"published_profile\": {\"ball\": \"new Penn ATP Extra Duty\", \"mass_kg\": 0.05803, \"radius_m\": 0.03284, \"air_density_kg_m3\": 1.21},\n"
        << "    \"fixture_conditions\": \"Zero spin/gravity; production viscosity retained; one native timestep; mass, radius and density read from compiled fixture\",\n"
        << "    \"engineering_test_tolerance\": 0.005,\n"
        << "    \"scope\": \"Parameterization to published mean; force implementation check, not held-out trajectory validation\",\n"
        << "    \"production_ball_configuration\": " << std::quoted(production_drag.configuration) << ",\n"
        << "    \"published_ball_configuration\": " << std::quoted(published_drag.configuration) << ",\n"
        << "    \"samples\": [\n";
    bool first = true, drag_pass = true;
    for (const bool published : {false, true}) for (const auto& sample : (published ? published_drag : production_drag).samples) {
        if (!first) out << ",\n";
        first = false;
        const bool pass = std::abs(sample.engineering_cd-published_cd) <= numerical_cd_tolerance;
        drag_pass = drag_pass && pass;
        out << "      {\"profile\": \"" << (published ? "published_ball" : "production_ball")
            << "\", \"axis\": " << sample.axis << ", \"speed_mps\": " << sample.speed
            << ", \"force_N\": " << sample.force << ", \"effective_cd\": " << sample.engineering_cd
            << ", \"within_numerical_target\": " << (pass ? "true" : "false") << '}';
    }
    out << "\n    ],\n    \"force_check_pass\": " << (drag_pass ? "true" : "false") << "\n  },\n"
        << "  \"production_spin_probe\": {\n    \"configuration\": " << std::quoted(production_drag.configuration) << ",\n"
        << "    \"scope\": \"Absolute one-tick delta-v/dt evidence at +Z 20 m/s and X spin; lift and spin decay are not empirically calibrated\",\n"
        << "    \"samples\": [\n";
    first=true;
    for (const auto& sample : spin_probe(production_xml)) {
        if (!first) out << ",\n";
        first=false;
        out << "      {\"incoming_spin_x_radps\": " << sample.incoming_spin << ", \"acceleration_mps2\": ";
        vector(out,sample.acceleration);
        out << ", \"outgoing_spin_radps\": "; vector(out,sample.outgoing_spin); out << '}';
    }
    out << "\n    ]\n  },\n  \"emery_bounce\": {\n    \"source\": \"https://doi.org/10.1119/1.1450571\",\n"
        << "    \"location\": \"Table I and Sections IV-VII\",\n"
        << "    \"scope\": \"Emery comparator, not production hardcourt calibration; production normal law unchanged\",\n"
        << "    \"boundary_approximations\": \"Fixed infinite plane replaces supported 3.8 kg timber platform; no air; paper approximate inertia, not measured inertia\",\n"
        << "    \"impulse_conventions\": \"Comparisons use the paper's gravity-neglecting momentum approximation; solver_contact_impulse_Ns excludes gravity and is reported separately\",\n"
        << "    \"mass_kg\": 0.057, \"radius_m\": 0.033, \"inertia_kg_m2\": 0.0000342,\n"
        << "    \"emery_friction\": 0.67, \"friction_reported_uncertainty\": 0.03,\n"
        << "    \"incident_velocity_mps\": [2.10,-0.77,0], \"incident_reported_uncertainty\": [0.04,0.014,0],\n"
        << "    \"spin_radps\": [0,0,0], \"cases\": [\n";
    first = true;
    for (const double timestep : {.000125, .0000625}) {
        const auto bounce = emery_bounce(production_xml,timestep);
        const auto& c = bounce.contact;
        if (!first) out << ",\n";
        first = false;
        out << "      {\"configuration\": " << std::quoted(bounce.configuration)
            << ", \"timestep_s\": " << timestep << ", \"gravity_mps2\": "; vector(out,bounce.gravity);
        out << ", \"contact_incoming_mps\": "; vector(out,c.incoming_velocity);
        out << ", \"contact_outgoing_mps\": "; vector(out,c.outgoing_velocity);
        out << ", \"outgoing_spin_radps\": "; vector(out,c.outgoing_spin);
        out << ", \"solver_contact_impulse_Ns\": "; vector(out,c.impulse);
        out << ", \"duration_s\": " << c.duration << ", \"penetration_m\": " << c.max_penetration
            << ", \"warnings\": " << bounce.warnings << ", \"comparisons\": [\n";
        bool empirical_pass = true, first_value = true;
        for (const auto& value : comparisons(bounce)) {
            if (!first_value) out << ",\n";
            first_value = false;
            const bool pass = std::abs(value.actual-value.expected) <= value.uncertainty;
            empirical_pass = empirical_pass && pass;
            out << "        {\"quantity\": \"" << value.name << "\", \"unit\": \"" << value.unit
                << "\", \"published\": " << value.expected << ", \"reported_uncertainty\": " << value.uncertainty
                << ", \"simulated\": " << value.actual << ", \"residual\": " << value.actual-value.expected
                << ", \"within_reported_interval\": " << (pass ? "true" : "false") << '}';
        }
        out << "\n      ], \"all_within_reported_intervals\": " << (empirical_pass ? "true" : "false") << '}';
    }
    out << "\n    ]\n  }\n}\n";
}
} // namespace physics_references
