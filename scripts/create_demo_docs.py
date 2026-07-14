from pathlib import Path

DOCS = {
    "bearing_fault_sop.md": """# Bearing Fault SOP

## Symptoms
- Abnormal increase in stand torque.
- Motor power rises together with torque under similar reduction.
- Rolling force may become unstable.
- In real plants, bearing housing temperature or vibration may increase.

## Likely causes
- Bearing wear.
- Lubrication starvation.
- Lubricant contamination.
- Coupling misalignment.
- Overload or excessive strip tension.

## Immediate actions
1. Reduce rolling load if production permits.
2. Inspect lubrication flow, pressure, and contamination.
3. Check bearing housing temperature.
4. Check coupling alignment and mechanical looseness.
5. Increase monitoring frequency for torque, motor power, and vibration.

## Planned maintenance
- Schedule bearing inspection during the current or next maintenance window.
- Reserve a bearing set if anomaly probability is high or repeated.
""",
    "electric_motor_fault_sop.md": """# Electric Motor Fault SOP

## Symptoms
- Motor power deviation is high.
- Power-to-torque relationship is inconsistent.
- Motor power increases without proportional mechanical load increase.
- Drive alarms, current imbalance, or cooling issues may appear in plant systems.

## Likely causes
- Motor efficiency degradation.
- Drive module fault.
- Electrical supply imbalance.
- Cooling problem.
- Mechanical overload reflected as secondary electrical stress.

## Immediate actions
1. Check drive alarms and motor current balance.
2. Inspect motor cooling and ventilation.
3. Compare torque increase against motor power increase.
4. If torque also rises, inspect mechanical sources before replacing motor parts.
""",
    "work_roll_friction_sop.md": """# Work Roll Friction SOP

## Symptoms
- Rolling force increases under stable reduction.
- Torque increases with force.
- Work roll mileage is high.
- Force-per-reduction ratio increases.

## Likely causes
- Roll surface wear.
- Poor lubrication or emulsion concentration.
- Excessive roll roughness.
- Roll cooling issue.

## Immediate actions
1. Check emulsion concentration and flow.
2. Inspect work roll surface condition.
3. Verify roll cooling nozzles.
4. Plan roll change if mileage threshold is crossed.
""",
    "reduction_scheme_anomaly_sop.md": """# Reduction Scheme Anomaly SOP

## Symptoms
- Mill-level abnormality across multiple stands.
- Reduction pattern changes unexpectedly.
- Rolling force and torque deviations appear in several stands.
- Gap and thickness reduction signals are inconsistent.

## Likely causes
- Incorrect pass schedule.
- Setup model mismatch.
- Gauge control issue.
- Material property mismatch.

## Immediate actions
1. Verify pass schedule and reduction setup.
2. Check entry and exit thickness values.
3. Compare roll gap settings against recipe.
4. Confirm material grade and yield-strength assumptions.
5. Escalate to process engineer if multiple stands are affected.
""",
    "maintenance_priority_policy.md": """# Maintenance Priority Policy

Priority score should consider:
- anomaly probability,
- fault confidence,
- health index,
- equipment criticality,
- spare availability,
- procurement lead time,
- repeated alert count,
- production impact.

Risk bands:
- 0 to 30: Low
- 31 to 55: Medium
- 56 to 75: High
- 76 to 100: Critical

Critical alerts require immediate inspection or controlled load reduction.
High alerts require same-shift maintenance planning and spare verification.
Medium alerts require increased monitoring and planned inspection.
""",

    "cascading_impact_sop.md": """# Cascading Impact SOP

## Context
A 5-stand tandem cold mill is a coupled process. A local abnormality in one stand can disturb inter-stand tension, adjacent motor load, exit gauge stability, and product quality.

## Symptoms
- Primary stand has high torque, force, or motor power deviation.
- Adjacent upstream/downstream tension shifts.
- Downstream motor power changes after upstream load disturbance.
- Multiple stands show moderate z-score deviations.

## Actions
1. Identify the primary abnormal stand.
2. Compare upstream and downstream stand torque/force/motor power.
3. Review inter-stand tension before isolating the fault as a single component issue.
4. Stabilize speed/load if cascading risk is high.
5. Document whether the adjacent stand effect is primary, secondary, or unrelated.
""",
    "physical_constraint_rules.md": """# Physical Constraint Rules for Rolling Mill Diagnosis

## Rule patterns
- Torque and motor power rising together under stable reduction suggests mechanical load increase.
- Motor power rising without proportional torque increase suggests electrical drive or motor efficiency issue.
- Force and torque rising with high work-roll mileage suggests friction, lubrication, or roll wear.
- Multi-stand gap/reduction deviation suggests reduction schedule or setup abnormality.
- Adjacent tension shifts suggest cascading line imbalance.

## Use
These rules validate ML predictions and help maintenance engineers choose the first inspection path. They should be used with SOPs, sensor evidence, and engineer judgement.
""",
    "emergency_shutdown_sop.md": """# Emergency Shutdown SOP

Emergency shutdown or controlled slowdown should be considered when:
- risk is critical,
- anomaly probability remains above threshold for repeated windows,
- torque and motor power rise rapidly together,
- product quality or operator safety may be affected,
- the same critical fault repeats after maintenance.

Before shutdown:
1. Notify shift supervisor.
2. Stabilize strip tension if possible.
3. Reduce speed/load in a controlled way.
4. Log the event and preserve sensor evidence.
""",
    "strip_tension_instability_sop.md": """# Strip Tension Instability SOP

## Symptoms
- Upstream and downstream stands show alternating torque or speed deviations.
- Exit thickness control becomes noisy during otherwise steady production.
- Tension-related alarms repeat around adjacent stands.

## Likely causes
- Cascading stand mismatch.
- Setup or speed synchronization issue.
- Roll friction change causing downstream load redistribution.
- Tension measurement or actuator drift.

## Immediate actions
1. Compare adjacent-stand torque, speed, and force trends in the same alarm window.
2. Check whether one stand is the primary disturbance source before adjusting multiple stands.
3. Stabilize speed or load gradually and avoid abrupt setpoint changes.
4. Validate tension feedback channel health if sensor drift is suspected.
""",
    "hydraulic_agc_servo_sop.md": """# Hydraulic AGC / Servo Response SOP

## Symptoms
- Gap or thickness correction appears delayed or oscillatory.
- Force rises while commanded gap adjustments fail to settle.
- Several short alarms appear during setup transitions.

## Likely causes
- Servo lag or valve sticking.
- Hydraulic pressure instability.
- Control tuning mismatch after grade or gauge change.
- Feedback sensor calibration offset.

## Immediate actions
1. Review commanded versus actual gap response during the alarm window.
2. Check hydraulic pressure stability and actuator response time.
3. Inspect for oscillation introduced by tuning or recent maintenance work.
4. Confirm gauge and gap feedback channels are healthy before replacing hardware.
""",
    "coolant_emulsion_management_sop.md": """# Coolant and Emulsion Management SOP

## Symptoms
- Rolling force and torque rise under similar reduction.
- Work-roll surface temperature or friction-related behaviour worsens.
- Surface quality issues appear together with abnormal stand load.

## Likely causes
- Low emulsion concentration.
- Inadequate flow or blocked nozzles.
- Temperature drift in coolant circuit.
- Contamination affecting lubrication performance.

## Immediate actions
1. Check emulsion concentration, conductivity, and return contamination.
2. Verify nozzle spray pattern and flow balance across the stand.
3. Inspect coolant temperature trend before changing mechanical components.
4. Log whether load normalized after lubrication correction.
""",
    "gearbox_vibration_sop.md": """# Gearbox Vibration and Backlash SOP

## Symptoms
- Torque fluctuations increase without matching force rise.
- Periodic vibration or mechanical noise is reported near a stand drive.
- Load spikes repeat after maintenance restart.

## Likely causes
- Gear wear or backlash growth.
- Coupling looseness.
- Lubrication breakdown in gearbox.
- Misalignment between motor and gearbox.

## Immediate actions
1. Compare vibration, torque, and power behaviour across restart windows.
2. Inspect coupling integrity and gearbox lubrication condition.
3. Check for abnormal noise, temperature rise, or repeated transient spikes.
4. Escalate for planned inspection if the pattern repeats under similar load.
""",
    "sensor_validation_instrumentation_sop.md": """# Sensor Validation and Instrumentation SOP

## Use this SOP when
- ML evidence suggests a fault but physical behaviour is inconsistent.
- Power, torque, force, gap, or tension channels disagree with each other.
- The same alarm disappears after instrumentation reset or maintenance.

## Checks
1. Compare the suspect channel against related process variables.
2. Review recent maintenance, calibration, or wiring changes.
3. Check whether the signal drift is isolated to one stand or repeated elsewhere.
4. Validate historian continuity and missing-data patterns.

## Decision note
If physical rules do not support the signal, treat instrumentation drift as a first inspection path before replacing mechanical parts.
""",
    "shift_handover_rca_template.md": """# Shift Handover RCA Template

## Summary fields
- Alarm ID
- Affected stand / asset
- Predicted fault and confidence
- Risk level and anomaly probability
- Main telemetry evidence
- SOP or checklist referenced
- Immediate action taken
- Open checks for next shift

## Handover guidance
Use concise, evidence-based language. Record whether the issue is primary to one stand or likely part of a cascading mill imbalance.
""",
    "preventive_maintenance_matrix.md": """# Preventive Maintenance Matrix for Tandem Cold Mill

## Daily checks
- Lubrication flow and contamination review.
- Drive cooling and ventilation review.
- Abnormal load, vibration, and tension trend scan.

## Weekly checks
- Roll condition and emulsion performance review.
- Coupling, gearbox, and bearing trend review.
- Instrumentation drift or calibration exception review.

## Trigger-based checks
- Any repeated high-risk Steel Pilot alarm should create a focused inspection plan.
- Any critical alarm with cascading risk should be reviewed before the next production campaign.
""",
    "stand_alignment_chock_inspection_sop.md": """# Stand Alignment and Chock Inspection SOP

## Symptoms
- Force, torque, and strip steering behaviour become unstable on one stand.
- Mechanical load rises repeatedly after roll change or mechanical work.
- Adjacent stands begin compensating for one stand's instability.

## Likely causes
- Chock looseness.
- Stand alignment drift.
- Roll seating issue after changeover.
- Mechanical binding in stand components.

## Immediate actions
1. Review whether the fault started after roll change or stand intervention.
2. Inspect seating, alignment, and looseness indicators before changing electrical components.
3. Confirm whether adjacent-stand load normalized after mechanical correction.
""",
    "drive_cooling_and_ventilation_sop.md": """# Drive Cooling and Ventilation SOP

## Symptoms
- Motor or drive-related alarms increase at higher production load.
- Power deviation rises without proportional torque increase.
- Alarm rate increases during hot ambient or reduced airflow periods.

## Immediate actions
1. Check cooling fan status, filter blockage, and airflow.
2. Compare drive temperature with load trend during the alarm window.
3. Confirm whether the issue is thermal/electrical before replacing mechanical parts.
4. Record whether abnormal power behavior improves after cooling correction.
""",
    "fault_signature_reference.md": """# Rolling Mill Fault Signature Reference

## Mechanical-load signature
- Torque and power rise together.
- Force often rises or becomes unstable.
- Bearing, friction, lubrication, or alignment checks should come first.

## Electrical-drive signature
- Power rises without proportional torque increase.
- Cooling, drive health, current balance, and supply quality checks should come first.

## Multi-stand setup signature
- Several stands show coordinated deviation.
- Reduction schedule, AGC response, and tension balance must be checked before isolating one component.
""",
    "alarm_response_playbook.md": """# Alarm Response Playbook

## First 5 minutes
1. Confirm affected alarm ID and active stand.
2. Review top telemetry evidence and risk level.
3. Check whether the issue is localized or cascading.
4. Pull the relevant SOP/checklist before acting.

## First 30 minutes
1. Record probable cause and uncertainty.
2. Execute immediate safe actions.
3. Assign planned checks for next shift if not fully resolved.
4. Update the logbook with evidence and action taken.
""",
}


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, content in DOCS.items():
        path = docs_dir / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(DOCS)} docs to {docs_dir.resolve()}")


if __name__ == "__main__":
    main()
