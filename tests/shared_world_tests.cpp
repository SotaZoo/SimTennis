#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
void check(bool value, const std::string& message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
std::string read(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(file), {}};
}
double length(tennis::Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
tennis::Vec3 difference(tennis::Vec3 a, tennis::Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
const tennis::BodyPose& body(const tennis::PlayerSnapshot& player, const std::string& name) {
    for (const auto& b : player.bodies) if (b.name == name) return b;
    throw std::runtime_error("Missing canonical player body " + name);
}
tennis::Vec3 rotate(tennis::Quaternion q, tennis::Vec3 v) {
    const tennis::Vec3 t{2*(q.y*v.z-q.z*v.y),2*(q.z*v.x-q.x*v.z),2*(q.x*v.y-q.y*v.x)};
    return {v.x+q.w*t.x+q.y*t.z-q.z*t.y,
            v.y+q.w*t.y+q.z*t.x-q.x*t.z,
            v.z+q.w*t.z+q.x*t.y-q.y*t.x};
}
void parked_ball(tennis::Simulation& sim) { sim.feed_ball({20,20,20},{0,0,0}); }
void command(tennis::Simulation& sim, int player, tennis::Vec3 movement,
             tennis::JointValues targets = tennis::JointValues(tennis::joint_count,0)) {
    check(sim.set_intent({sim.snapshot().time,targets,1,movement},player), "fresh indexed intent is accepted");
}

void model_and_compatibility(const std::string& fixed, const std::string& match) {
    tennis::Simulation legacy(fixed);
    auto old = legacy.snapshot();
    check(old.players.size()==1 && !old.players[0].movable, "legacy supported fixture keeps one fixed player");
    check(old.joint_angles==old.players[0].joint_angles, "legacy arm fields alias player zero");
    check(legacy.set_intent({0,tennis::JointValues(tennis::joint_count,0),1}), "explicit arm targets and default-player API remain valid");
    check(!legacy.set_intent({0,tennis::JointValues(tennis::joint_count,0),1},1), "missing player is rejected");
    tennis::Simulation sim(match);
    auto s=sim.snapshot();
    check(s.players.size()==2, "match has two players in one simulation");
    check(std::count_if(s.bodies.begin(),s.bodies.end(),[](const auto& b){return b.name=="ball";})==1,
          "shared snapshot contains exactly one ball");
    for (const auto& player:s.players) {
        check(player.movable && player.bodies.size()==5, "each movable player has five canonical bodies");
        for (const char* name:{"torso","upper_arm","forearm","hand","racket"}) (void)body(player,name);
    }
    check(s.players[0].torso_position.z>0 && s.players[1].torso_position.z<0, "players start on opposite halves");
    check(!sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1},2) && !sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1},-1), "invalid player indices cannot address other state");
    check(!sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1,{0,1,0}},0), "vertical movement is rejected by planar interface");
    check(!sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1,{std::numeric_limits<double>::quiet_NaN(),0,0}},1), "nonfinite movement is rejected");
}

void movement_and_expiry(const std::string& match) {
    tennis::Simulation sim(match); parked_ball(sim);
    const auto initial=sim.snapshot();
    command(sim,0,{2,0,0}); command(sim,1,{2,0,0}); sim.step(1);
    auto first=sim.snapshot();
    for (int p=0;p<2;++p) {
        check(length(first.players[p].movement_effort)<=700.000001,"movement force has a resultant cap");
        check(first.players[p].torso_velocity.x>0 && first.players[p].torso_velocity.x<.01,"both sides accelerate in world +X without velocity writes");
        check(length(difference(first.players[p].torso_position,initial.players[p].torso_position))<.00001,"first tick does not teleport the support");
    }
    for(int frame=0;frame<26;++frame) {command(sim,0,{2,0,0});sim.step(80);}
    auto expired=sim.snapshot();
    check(!expired.players[0].input_expired && expired.players[1].input_expired,"input expiration is independent for each player");
    for(int frame=0;frame<50;++frame) {command(sim,0,{2,0,0});sim.step(80);}
    auto later=sim.snapshot();
    check(later.players[0].torso_velocity.x>1.7 && std::abs(later.players[1].torso_velocity.x)<.2,"expired movement brakes while other player keeps moving");
    check(later.players[0].torso_position.x>initial.players[0].torso_position.x+.7,"support traverses the court physically");
    check(!sim.set_intent({0,tennis::JointValues(tennis::joint_count,0),1,{2,0,0}},0),"old movement observations are rejected");
    check(later.diagnostics.warnings==0,"ordinary movement has no solver warnings");
}

void mirror_and_bounds(const std::string& match) {
    tennis::Simulation sim(match); parked_ball(sim);
    const tennis::JointValues arm{.1,-.1,.2,.05,.03,-.04};
    for(int frame=0;frame<100;++frame) {
        command(sim,0,{1.2,0,-.8},arm);command(sim,1,{-1.2,0,.8},arm);sim.step(80);
    }
    auto s=sim.snapshot();
    for(const char* name:{"torso","upper_arm","forearm","hand","racket"}) {
        const auto a=body(s.players[0],name).position,b=body(s.players[1],name).position;
        check(length({a.x+b.x,a.y-b.y,a.z+b.z})<1e-8,"180-degree mirrored movement and arm mechanics remain symmetric");
    }
    double overshoot=0, peak_speed=0, peak_force=0;
    for(int direction:{-1,1}) {
        sim.reset();parked_ball(sim);
        for(int frame=0;frame<700;++frame) {
            command(sim,0,{20.*direction,0,20.*direction});
            command(sim,1,{-20.*direction,0,-20.*direction});
            sim.step(80); s=sim.snapshot();
            for(int p=0;p<2;++p) {
                const auto& player=s.players[p];
                peak_speed=std::max(peak_speed,length(player.torso_velocity));
                peak_force=std::max(peak_force,length(player.movement_effort));
                const double z=p ? -player.torso_position.z : player.torso_position.z;
                overshoot=std::max({overshoot,std::abs(player.torso_position.x)-5.2,.65-z,z-12.885});
                check(length(player.movement_target)<=5.000001,"requested movement is bounded to 5 m/s in the plane");
            }
        }
        check(s.diagnostics.warnings==0,"sustained movement at half-court limits has no solver warning");
    }
    check(peak_speed<=5.1,"unloaded movement remains within the declared 5.1 m/s numerical envelope");
    check(peak_force<=700.000001,"diagonal commands obey the resultant force cap");
    check(overshoot<=.02,"support remains within a declared 2 cm soft-stop allowance");
    std::cout<<"MOVEMENT_ENVELOPE peak_speed="<<peak_speed<<" peak_force="<<peak_force<<" max_bounds_overshoot="<<overshoot<<'\n';
}

void replay(const std::string& match) {
    tennis::Simulation sim(match);parked_ball(sim);
    command(sim,0,{2,0,-1},{.2,-.1,.3,.1,.1,.1});
    command(sim,1,{-1,0,2},{-.1,.1,.4,-.1,-.1,-.1});sim.step(1000);
    const auto saved=sim.capture();
    sim.step(1700);const auto a=sim.snapshot();
    sim.restore(saved);for(int i=0;i<1700;++i)sim.step(1);const auto b=sim.snapshot();
    check(a.time==b.time && a.tick==b.tick,"shared checkpoint preserves time and tick");
    for(int p=0;p<2;++p) {
        check(a.players[p].joint_angles==b.players[p].joint_angles && a.players[p].controller_targets==b.players[p].controller_targets,"both arm histories replay exactly across step chunks");
        check(length(difference(a.players[p].torso_position,b.players[p].torso_position))==0 && length(difference(a.players[p].torso_velocity,b.players[p].torso_velocity))==0,"both moving supports replay exactly");
        check(a.players[p].input_expired==b.players[p].input_expired,"independent expiration survives restore");
    }
    sim.feed_ball({0,3,0},{0,0,0});const auto fed=sim.snapshot();
    check(length(difference(fed.players[0].torso_position,b.players[0].torso_position))==0 && fed.players[1].controller_targets==b.players[1].controller_targets,"scenario feed preserves both players and controller histories");
    sim.reset();const auto reset=sim.snapshot();
    check(reset.players[0].input_expired && reset.players[1].input_expired && length(reset.players[0].torso_velocity)==0 && length(reset.players[1].torso_velocity)==0,"reset clears both movement inputs and velocities");
}

tennis::Snapshot impact(const std::string& xml,int player) {
    tennis::Simulation sim(xml);
    const auto racket=body(sim.snapshot().players[player],"racket");
    const auto offset=rotate(racket.orientation,{.43,0,0});
    const auto normal=rotate(racket.orientation,{0,0,1});
    sim.feed_ball({racket.position.x+offset.x-.3*normal.x,racket.position.y+offset.y-.3*normal.y,racket.position.z+offset.z-.3*normal.z},
                  {10*normal.x,10*normal.y,10*normal.z});
    sim.step(static_cast<std::uint64_t>(std::llround(.12/sim.snapshot().timestep)));
    return sim.snapshot();
}
void contacts_and_refinement(const std::string& match) {
    auto fine=match;
    const auto at=fine.find("timestep=\"0.000125\"");
    if(at==std::string::npos)throw std::runtime_error("Missing match timestep");
    fine.replace(at,std::string("timestep=\"0.000125\"").size(),"timestep=\"0.0000625\"");
    for(int player=0;player<2;++player) {
        const auto a=impact(match,player),b=impact(fine,player);
        check(a.diagnostics.racket_episodes==1 && b.diagnostics.racket_episodes==1,"shared ball hits one selected physical racket");
        auto event=std::find_if(a.contacts.begin(),a.contacts.end(),[](const auto& c){return c.surface=="racket";});
        check(event!=a.contacts.end() && event->complete && event->player==player,"completed contact identifies the actual racket owner");
        check(a.players[player].racket_episodes==1 && a.players[1-player].racket_episodes==0,"racket contact counters are player-specific");
        check(length(difference(a.ball_velocity,b.ball_velocity))<=std::max(.01,.01*length(b.ball_velocity)),"shared-world impact velocity converges at half timestep");
        check(length(difference(a.ball_spin,b.ball_spin))<=std::max(1.,.01*length(b.ball_spin)),"shared-world impact spin converges at half timestep");
        check(a.diagnostics.warnings==0 && b.diagnostics.warnings==0,"shared impacts emit no solver warnings");
        std::cout<<"SHARED_IMPACT player="<<player<<" delta_velocity="<<length(difference(a.ball_velocity,b.ball_velocity))<<" delta_spin="<<length(difference(a.ball_spin,b.ball_spin))<<'\n';
    }
    tennis::Simulation court(match);court.feed_ball({0,.3,5},{0,-3,0});court.step(1500);
    for(const auto& event:court.snapshot().contacts)check(event.surface=="racket" || event.player==-1,"court/net contacts have no racket owner");
}
} // namespace

int main(int argc,char** argv) {
    try {
        const std::string path=argc>1?argv[1]:"models";
        const auto fixed=read(path+"/supported_arm.xml"),match=read(path+"/match.xml");
        model_and_compatibility(fixed,match);movement_and_expiry(match);mirror_and_bounds(match);replay(match);contacts_and_refinement(match);
    } catch(const std::exception& e) {check(false,e.what());}
    std::cout<<(failures?"FAIL":"PASS")<<": "<<failures<<" shared-world behavioral checks failed\n";
    return failures?1:0;
}
