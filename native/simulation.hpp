#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tennis {

// Physical data use metres, seconds, kilograms and radians in a right-handed,
// Y-up frame. Quaternions are wxyz. Returned snapshots own all of their data.
struct Vec3 {
    double x = 0, y = 0, z = 0;
};
struct Quaternion {
    double w = 1, x = 0, y = 0, z = 0;
};
struct BodyPose {
    std::string name;
    Vec3 position;
    Quaternion orientation;
};

// Legacy arm order, retained for the recorded-input drill. Other models declare
// their controller axes in MJCF; snapshots expose that order by name.
inline constexpr std::size_t joint_count = 6;
using JointValues = std::vector<double>;
inline constexpr std::array<const char*, joint_count> joint_names{
    "shoulder_yaw", "shoulder_pitch", "elbow_flexion",
    "shoulder_axial", "forearm_roll", "wrist_bend"};

struct Intent {
    // Simulation-clock observation time, monotonically increasing per reset.
    // Missing/low-confidence/older-than-250ms observations are rejected. The
    // last accepted observation expires after 250ms; control then holds neutral.
    double time = 0;
    JointValues targets = JointValues(joint_count, 0); // model axis order, radians
    double confidence = 1;
    Vec3 movement_velocity{}; // WORLD X/Z m/s; Y must be zero; expires with the arm intent
};

// Direct actuator commands for learned control. Units are N.m and world N.
// Held until replaced or expired after 250 ms; expiration applies zero effort.
struct Effort {
    double time = 0;
    JointValues torques{};
    Vec3 movement_force{};
};

struct ContactEvent {
    std::string surface; // racket, court or net
    int player = -1; // racket owner 0/1; court and net have no owner
    std::uint64_t first_tick = 0;
    std::uint64_t last_tick = 0;
    Vec3 position; // world ball centre at the first positive-force contact
    Vec3 impulse; // integrated contact force on the ball, world N.s
    double max_penetration = 0;
    double duration = 0;
    Vec3 incoming_velocity;
    Vec3 outgoing_velocity;
    Vec3 incoming_spin;
    Vec3 outgoing_spin;
    bool complete = false;
};

struct Diagnostics {
    std::uint64_t racket_episodes = 0;
    std::uint64_t court_episodes = 0;
    std::uint64_t net_episodes = 0;
    // Each counts at most once per physics tick, regardless of affected axes.
    std::uint64_t target_clipped_ticks = 0;
    std::uint64_t torque_saturated_ticks = 0;
    std::uint64_t rate_limited_ticks = 0;
    std::uint64_t movement_target_clipped_ticks = 0;
    std::uint64_t movement_force_saturated_ticks = 0;
    std::uint64_t rejected_inputs = 0;
    std::uint64_t stale_inputs = 0;
    std::uint64_t warnings = 0;
    double kinetic_energy = 0;
    double potential_energy = 0;
    double actuator_work = 0;
    double max_penetration = 0;
    double max_racket_speed = 0; // centre of mass, sampled at every tick boundary
    double max_stringbed_speed = 0; // stringbed geometry centre, same sampling
    bool input_expired = true;
    bool faulted = false; // requires reset or a known-good checkpoint restore
};

struct PlayerSnapshot {
    // Names are canonical for each player, independent of model namespaces.
    std::vector<BodyPose> bodies;
    std::vector<std::string> joint_names;
    JointValues joint_angles{};
    JointValues joint_velocities{};
    JointValues actuator_efforts{};
    JointValues controller_targets{};
    Vec3 torso_position;
    Vec3 torso_velocity; // world velocity at the support origin
    Vec3 base_position; // model's first declared body; not necessarily a torso
    Vec3 base_velocity;
    Vec3 movement_effort; // world planar force, N
    Vec3 movement_target; // bounded world velocity target, m/s
    Vec3 racket_velocity;
    Vec3 racket_spin;
    std::uint64_t racket_episodes = 0;
    bool input_expired = true;
    bool movable = false;
};

struct Snapshot {
    std::uint64_t tick = 0;
    double time = 0;
    double timestep = 0;
    std::vector<BodyPose> bodies;
    Vec3 ball_velocity;
    Vec3 ball_spin;
    Vec3 racket_velocity; // world centre-of-mass velocity (body pose is the grip)
    Vec3 racket_spin;
    JointValues joint_angles{};
    JointValues actuator_efforts{};
    JointValues controller_targets{}; // bounded/rate-limited setpoints at this tick
    Diagnostics diagnostics;
    std::vector<PlayerSnapshot> players;
    // Most recent 32 episodes, including current unfinished episodes.
    std::vector<ContactEvent> contacts;
};

namespace detail { struct SavedState; }

// An opaque, owned same-model checkpoint, including MuJoCo integration state,
// command history, controller state, accumulated events and diagnostics.
struct CapturedState {
    std::shared_ptr<const detail::SavedState> state;
};

class Simulation {
public:
    explicit Simulation(const std::string& model_xml);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void reset();
    // Place a movable support within its slide limits before the first tick.
    // Does not change the ball or racket joints; never usable during an attempt.
    void set_player_start(Vec3 position, int player = 0);
    // Explicit scenario setup. This is never called to award/repair a hit.
    // It clears prior impact episodes but does not move or energize the racket.
    void feed_ball(Vec3 position, Vec3 velocity, Vec3 spin = {});
    bool set_intent(const Intent& intent, int player = 0);
    bool set_effort(const Effort& effort, int player = 0);
    void step(std::uint64_t ticks);
    Snapshot snapshot() const;
    CapturedState capture() const;
    void restore(const CapturedState& state);
    // Initialize an isolated presentation predictor from authoritative player
    // observations. Resets the world and parks the ball; never use for matches.
    void reset_prediction(int player, const PlayerSnapshot& source);
    std::string configuration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tennis
