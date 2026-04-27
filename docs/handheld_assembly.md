# Handheld Assembly Guide

## Scope
This guide covers the physical assembly workflow for the Pi handheld prototype used by the PQ monitor runtime.

## Required Hardware
- Raspberry Pi 5 (8 GB)
- Official Raspberry Pi display with touch
- Active cooling (Pi 5 fan or equivalent)
- Stable 5V power source sized for Pi 5 peak load
- Teensy 4.1 sensing node with isolated analog front-end
- USB cable from Pi to Teensy
- Non-conductive enclosure with separated HV and LV zones

## Tools
- Insulated screwdriver set
- Wire ferrules and crimp tool
- Cable ties and adhesive mounts
- Multimeter for continuity and grounding checks

## Assembly Steps
1. Mount Pi 5 and display in enclosure standoffs.
2. Install active cooling and verify airflow path is not blocked.
3. Route display ribbon and touch interface cable away from power wiring.
4. Mount Teensy board in a dedicated low-voltage compartment.
5. Route USB cable from Pi to Teensy with strain relief.
6. Keep high-voltage sensing board compartment physically separated from Pi and USB wiring.
7. Bond enclosure ground as required by your lab safety policy.
8. Close enclosure only after continuity checks are complete.

## Wiring Separation Rules
- Never route mains-side conductors in the same bundle as USB, GPIO, or display wiring.
- Keep at least one physical barrier between HV sensing and LV compute compartments.
- Maintain isolation barrier integrity for AMC1301 and isolated supply paths.
- Do not connect isolated high-side ground to Pi ground.

## Thermal and Power Checklist
- Fan starts automatically at boot.
- CPU temperature remains below thermal throttle threshold during 30-minute run.
- PSU cable and connector remain cool to touch under sustained load.
- No display brown-out, USB reset, or service restart events in logs.

## Safety Checklist Before First Power-On
- Teensy and Pi grounds verified only on low-voltage side.
- Mains-side wiring inspected by qualified supervisor.
- Enclosure closed; no exposed HV terminals.
- Emergency power cut-off available at bench.

## Acceptance
Assembly is accepted when:
1. Device boots to dashboard without manual intervention.
2. Teensy stream reconnects automatically after cable reseat.
3. 30-minute monitoring run completes with no thermal shutdown or UI freeze.
