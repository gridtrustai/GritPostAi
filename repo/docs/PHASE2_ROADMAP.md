# Phase 2 roadmap (to 12 December 2026)

Priorities, in order of impact on the challenge goals.

1. **Measured baseline.** Replace the synthetic load with real consumption data from a public source (UCI ElectricityLoadDiagrams20112014), or STEG open data if available, scaled to Tunisian household profiles. Keep the same injection code and labels. Split the data in time into tuning and held-out test periods.
2. **Earlier detection of slow faults.** Add CUSUM on the profile residual and peer comparison between meters with similar profiles, to cut the drift delay (≈10 h) and the falsification delay (≈27 h).
3. **Explainability.** SHAP values for the Isolation Forest features, shown next to the rule evidence; compare with the current "main driver" heuristic.
4. **Calibrated confidence.** Calibrate the evidence score against labelled and operator-reviewed incidents (reliability plot), or keep calling it a score.
5. **Operator interface.** A small web app (for example Streamlit) listing incidents by route, showing evidence and charts, with accept, reject or override plus a reason, written back to the decision log. Track override rate and time per decision (over- and under-trust metrics).
6. **Collective defense demo.** Simulate 3 nodes exchanging pattern signatures (see `SHARED_LEARNING_DESIGN.md`) and measure whether a pattern learned in one node is detected sooner in another.
7. **Streaming mode.** Process readings one by one (for example over MQTT) to measure real response latency.
8. **Business scope.** Budget, implementation and deployment cost, continuous retraining plan, and integration with utility systems.
