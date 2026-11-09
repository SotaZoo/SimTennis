#include "simulation.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(bool good, const char* reason) { if (!good) throw std::runtime_error(reason); }
std::string replace(std::string xml,const std::string& before,const std::string& after) {
    const auto at=xml.find(before); check(at!=std::string::npos,"Fixture replacement must exist");
    xml.replace(at,before.size(),after); return xml;
}
void rejects(const std::string& xml) {
    try { tennis::Simulation invalid(xml); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Malformed model controller was accepted");
}
}

int main(int argc,char** argv) {
    try {
        check(argc==2,"Expected car model path");
        std::ifstream file(argv[1]); check(static_cast<bool>(file),"Car model must exist");
        const std::string xml{std::istreambuf_iterator<char>(file),{}};
        tennis::Simulation sim(xml);
        const auto initial=sim.snapshot();
        check(initial.players.size()==2,"Car scene must have two independent players");
        for (const auto& player:initial.players) {
            check(player.joint_names==std::vector<std::string>{"swing","tilt","lift"},"Model axis order must be explicit");
            check(player.joint_angles.size()==3 && player.joint_velocities.size()==3,"Model-sized position and speed feedback");
            check(player.bodies.front().name=="chassis","First declared body is the base");
        }
        check(!sim.set_intent({0,{0,0,0,0,0,0},1}),"Wrong mechanism commands must reject, not pad or truncate");
        check(!sim.set_intent({0,{},1}),"Empty command must reject");
        for (int frame=0;frame<100;++frame) {
            const auto state=sim.snapshot();
            for (int p=0;p<2;++p)
                check(sim.set_intent({state.time,{.2,.25,0},1,{p ? -2. : 2.,0,0}},p),"Car intent must accept");
            sim.step(64);
            const auto now=sim.snapshot();
            for (const auto& player:now.players) {
                check(std::abs(player.actuator_efforts[0])<=40 && std::abs(player.actuator_efforts[1])<=6 && std::abs(player.actuator_efforts[2])<=40,"Finite hinge torques");
                check(std::hypot(player.movement_effort.x,player.movement_effort.z)<=700.00001,"Finite resultant support force");
                for (double speed:player.joint_velocities) check(std::isfinite(speed),"Finite physical speeds");
            }
        }
        const auto moved=sim.snapshot();
        tennis::Simulation predicted(xml);
        for (int player=0;player<2;++player) {
            const auto before=sim.capture();
            const auto source=sim.snapshot().players[player];
            predicted.reset_prediction(player,source);
            for (int frame=0;frame<25;++frame) {
                const tennis::Vec3 velocity{player ? 2. : -2.,0,1};
                check(sim.set_intent({sim.snapshot().time,{.2,.25,0},1,velocity},player),"Authority movement accepts");
                check(predicted.set_intent({predicted.snapshot().time,{.2,.25,0},1,velocity},player),"Predicted movement accepts");
                sim.step(64); predicted.step(64);
            }
            const auto actual=sim.snapshot().players[player];
            const auto estimate=predicted.snapshot().players[player];
            check(std::hypot(actual.base_position.x-estimate.base_position.x,actual.base_position.z-estimate.base_position.z)<.0001,
                  "Snapshot-seeded movement reuses physical acceleration and braking");
            auto invalid=source;
            invalid.joint_velocities[0]=NAN;
            bool rejected=false;
            try { predicted.reset_prediction(player,invalid); } catch(const std::invalid_argument&) { rejected=true; }
            check(rejected,"Prediction rejects nonfinite state before resetting");
            sim.restore(before);
        }
        check(moved.players[0].base_position.x>initial.players[0].base_position.x+.2,"Near car moves through force");
        check(moved.players[1].base_position.x<initial.players[1].base_position.x-.2,"Mirrored car uses world movement");
        const auto checkpoint=sim.capture(); sim.step(2048); const auto next=sim.snapshot();
        sim.restore(checkpoint); for(int i=0;i<32;++i) sim.step(64);
        const auto replay=sim.snapshot();
        for(int p=0;p<2;++p)
            check(next.players[p].joint_angles==replay.players[p].joint_angles
                && next.players[p].joint_velocities==replay.players[p].joint_velocities,"Model-sized checkpoint must replay exactly");
        check(replay.diagnostics.warnings==0,"No numerical warnings");
        sim.reset();
        for (int frame=0;frame<100;++frame) {
            const auto state=sim.snapshot();
            check(sim.set_intent({state.time,{0,0,0},1,{4.5,0,0}},0),"Accelerate the car through bounded input");
            sim.step(64);
        }
        const auto released=sim.snapshot();
        check(released.players[0].base_velocity.x>4.4,"Braking trial starts at playing speed");
        for (int frame=0;frame<60;++frame) {
            check(sim.set_intent({sim.snapshot().time,{0,0,0},1,{}},0),"Release movement through normal input");
            sim.step(64);
        }
        const auto stopped=sim.snapshot();
        check(stopped.players[0].base_position.x-released.players[0].base_position.x<.45,
              "The car fixture brakes within 45 cm after key release");
        check(std::abs(stopped.players[0].base_velocity.x)<.2,"Released car settles without a position reset");
        rejects(replace(xml,"data=\"swing tilt lift\"","data=\"swing swing lift\""));
        rejects(replace(xml,"data=\"180 30 180\"","data=\"180\""));
        rejects(replace(xml,"data=\"16 6 16\"","data=\"16 -6 16\""));
        rejects(replace(xml,"<text name=\"controller_joints\" data=\"swing tilt lift\" />",""));
        rejects(replace(xml,"name=\"swing_motor\" joint=\"swing\" ctrlrange=\"-40 40\"","name=\"swing_motor\" joint=\"swing\" ctrlrange=\"-inf inf\""));
        rejects(replace(xml,"<motor name=\"swing_motor\"","<position kp=\"180\" name=\"swing_motor\""));
        rejects(replace(xml,"<motor name=\"swing_motor\"","<motor forcerange=\"-.001 .001\" name=\"swing_motor\""));
        rejects(replace(xml,"<motor name=\"movement_x_motor\"","<position kp=\"1000\" name=\"movement_x_motor\""));
        rejects(replace(xml,"data=\"chassis boom grip racket\"","data=\"boom chassis grip racket\""));
        std::cout<<"PASS: model-defined car axes, bounded independent movement, recovery feedback, checkpoints and invalid model rejection\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
