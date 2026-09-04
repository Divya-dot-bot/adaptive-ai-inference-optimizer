# Adaptive AI Inference Optimizer

A simulation-based research project that investigates how different autoscaling and infrastructure provisioning strategies affect the cost, latency, throughput, and SLA performance of an AI inference-serving system under changing workloads.

> **Important:** This project uses synthetically generated workloads. It does not use external datasets, cloud infrastructure, or real production traffic data. All infrastructure behavior and experimental results are simulated locally unless explicitly stated otherwise.

## Project Goals

The project simulates an AI inference-serving system where incoming requests change over time.

The system will:

* Generate realistic and reproducible synthetic AI inference workloads.
* Model baseline traffic, daily patterns, weekly patterns, peak hours, random noise, traffic spikes, and sudden workload surges.
* Forecast future inference workload.
* Simulate AI inference-serving infrastructure locally.
* Compare multiple autoscaling strategies.
* Optimize the trade-off between infrastructure cost, latency, throughput, and SLA violations.
* Simulate on-demand, spot-instance, and hybrid provisioning strategies.
* Run reproducible experiments under different workload conditions.

## Research Question

> To what extent can forecast-based and optimization-based autoscaling reduce the simulated infrastructure cost of an AI inference-serving system while maintaining throughput and meeting predefined latency and SLA requirements, compared with static and reactive autoscaling, under changing synthetic workloads and simulated spot-instance interruptions?

## Autoscaling Strategies

The project will compare:

1. **Static Provisioning**
   A fixed number of instances remains active regardless of workload changes.

2. **Reactive Autoscaling**
   Infrastructure scales based on current system metrics such as utilization or queue length.

3. **Forecast-Based Autoscaling**
   Future workload predictions are used to proactively determine infrastructure capacity.

4. **Optimization-Based Autoscaling**
   The number of active instances is selected by minimizing an objective that balances cost, SLA violations, and excessive latency.

## Forecasting Approaches

The project will compare:

1. Naive baseline forecasting.
2. Moving-average forecasting.
3. Machine learning forecasting using lag and time-based features.
4. An optional lightweight deep learning model only if experimentally justified and practical.

Forecasting performance will be evaluated using:

* MAE
* RMSE
* MAPE, where appropriate

## Simulation Metrics

The infrastructure simulator and experiments will measure:

* Total simulated infrastructure cost
* Average latency
* p95 latency, where practical
* Throughput
* Queue length
* Active instances
* SLA violations
* Forecasting error
* Infrastructure utilization
* Spot interruption and recovery effects

## Synthetic Workload

No external dataset is required.

All workload data is generated locally by the project using configurable synthetic patterns, including:

* Normal baseline traffic
* Daily seasonality
* Weekly seasonality
* Peak-hour behavior
* Random noise
* Random traffic spikes
* Sudden workload surges
* Changing traffic intensity
* Optional AI model types with different simulated inference costs

A fixed random seed can be used to reproduce experiments.

> Synthetic workloads represent simulated experimental conditions and must not be interpreted as real production AI traffic.

## Project Architecture

```text
Synthetic Workload Generator
            ↓
Historical Workload Data
            ↓
Forecasting Models
            ↓
Inference Infrastructure Simulator
            ↓
Autoscaling Decision Engine
            ↓
Optimization Module
            ↓
Performance Evaluation
            ↓
Results and Visualization
```

## Planned Project Structure

```text
adaptive-inference-optimizer/
│
├── data/
│   └── generated/
│       ├── workloads/
│       └── metadata/
│
├── notebooks/
│
├── src/
│   ├── workload/
│   ├── forecasting/
│   ├── simulation/
│   ├── autoscaling/
│   ├── optimization/
│   ├── spot/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│
├── configs/
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── experiment_logs/
│
├── tests/
├── app/
├── docs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Local-Only Design

This project is designed to run completely on a local laptop.

It does not require:

* AWS
* GCP
* Azure
* Cloud accounts
* Paid APIs
* Kubernetes
* Distributed infrastructure
* External datasets

Infrastructure behavior is modeled through a local Python simulation.

## Reproducibility

Experiments will use:

* Configurable random seeds
* Version-controlled source code
* Saved configuration files
* Locally generated datasets
* Documented simulation assumptions
* Controlled experimental scenarios

The same configuration and random seed should produce reproducible synthetic workloads and experimental conditions.

## Research Integrity

This is a **simulation-based research project**.

The project distinguishes between:

### Simulated assumptions

Examples include:

* Instance capacity
* Infrastructure pricing
* Latency model parameters
* SLA thresholds
* Spot interruption probabilities

### Synthetic data

Workload data generated locally by the project's workload generator.

### Experimental results

Metrics actually produced by running experiments using the implemented simulator.

The project will not present simulated results as measurements from a real production system.

## Development Status

The project is currently under development.

### Planned phases

* [x] Phase 1: Environment and project foundation
* [ ] Phase 2: Synthetic workload generator
* [ ] Phase 3: Workload analysis and scenario validation
* [ ] Phase 4: Baseline forecasting
* [ ] Phase 5: Machine learning forecasting
* [ ] Phase 6: Inference infrastructure simulator
* [ ] Phase 7: Static and reactive autoscaling
* [ ] Phase 8: Forecast-based autoscaling
* [ ] Phase 9: Optimization-based autoscaling
* [ ] Phase 10: Spot-instance and hybrid simulation
* [ ] Phase 11: Controlled experiments and evaluation
* [ ] Phase 12: Results and visualization
* [ ] Phase 13: Documentation and GitHub preparation

## License

This project is currently intended for educational and research purposes.
