# Evolutionary Architecture Search - Literature Review

**Date:** 2025-10-17
**Author:** Claude Code (Automated Research)
**Status:** Initial Review

## Overview

This document reviews key research in neural architecture search (NAS), evolutionary algorithms, and multi-objective optimization relevant to diffusion model architecture search.

## Key Research Areas

### 1. Neural Architecture Search (NAS)

#### Foundational Work

**DARTS: Differentiable Architecture Search**
- *Authors:* Liu et al. (2019)
- *Key Contribution:* Continuous relaxation of discrete architecture search space
- *Relevance:* Provides gradient-based alternative to evolutionary methods
- *Limitations:* High memory requirements, architecture collapse issues

**ENAS: Efficient Neural Architecture Search**
- *Authors:* Pham et al. (2018)
- *Key Contribution:* Parameter sharing across child models
- *Relevance:* Reduces computational cost of architecture search
- *Application:* Can be combined with evolutionary methods for efficiency

#### Evolutionary NAS

**AmoebaNet: Regularized Evolution for Image Classifier**
- *Authors:* Real et al. (2019)
- *Key Contribution:* Tournament selection with aging mechanism
- *Relevance:* Direct application to our evolutionary search
- *Implementation Notes:*
  - Population size: 100-200
  - Tournament size: 25
  - Aging: Remove oldest 25% of population
  - Mutation rate: Single mutation per genome

**NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm**
- *Authors:* Lu et al. (2019)
- *Key Contribution:* Multi-objective optimization for accuracy-efficiency trade-offs
- *Relevance:* Framework for Pareto front exploration
- *Objectives:* Accuracy, FLOPs, latency, memory

### 2. Multi-Objective Optimization

#### NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- *Authors:* Deb et al. (2002)
- *Key Features:*
  - Fast non-dominated sorting (O(MN²))
  - Crowding distance for diversity
  - Elite preservation
- *Implementation:* Use `pymoo` library
- *Parameters:*
  - Population size: 50-100
  - Crossover probability: 0.9
  - Mutation probability: 1/n (n = genome length)

#### NSGA-III (Reference Point Based)
- *Authors:* Deb & Jain (2014)
- *Advantages:* Better performance for >3 objectives
- *Relevance:* Quality, speed, memory, energy = 4 objectives
- *Trade-off:* More complex than NSGA-II

### 3. Diffusion Models

#### Architecture Innovations

**Latent Diffusion Models (Stable Diffusion)**
- *Authors:* Rombach et al. (2022)
- *Key Architecture:* VAE encoder/decoder + UNet diffusion model
- *Search Space Components:*
  - VAE configuration (channels, layers)
  - UNet depth and width
  - Attention block placement
  - Cross-attention mechanisms

**Consistency Models**
- *Authors:* Song et al. (2023)
- *Relevance:* Fast sampling without multi-step diffusion
- *Search Application:* Optimize for consistency training

### 4. Hardware-Aware NAS

**ProxylessNAS**
- *Authors:* Cai et al. (2019)
- *Key Contribution:* Direct optimization on target hardware
- *Relevance:* Essential for Apple Silicon optimization
- *Metrics:* Latency, memory, energy consumption

**FBNet: Hardware-Aware Efficient ConvNet Design**
- *Authors:* Wu et al. (2019)
- *Key Contribution:* Latency-aware differentiable search
- *Application:* Incorporate Apple Silicon latency models

#### Apple Silicon Considerations

**Unified Memory Architecture (UMA)**
- Zero-copy memory access between CPU/GPU
- Optimization: Minimize memory allocations
- Trade-off: Larger models possible vs thermal constraints

**Metal Performance Shaders (MPS)**
- Optimized kernels for Apple GPU
- Search Constraint: Prefer MPS-friendly operations
- Avoid: Non-contiguous tensors, dynamic shapes

**ANE (Apple Neural Engine)**
- Inference-optimized hardware
- Constraints: 16-bit precision, specific operation support
- Opportunity: Ultra-fast inference for compatible architectures

## Relevant Techniques

### Genetic Operators

#### Crossover Strategies
1. **One-point crossover:** Split architectures at random layer
2. **Uniform crossover:** Randomly select layers from each parent
3. **Block-wise crossover:** Exchange architectural blocks (encoder, decoder, attention)

#### Mutation Strategies
1. **Layer-level mutations:**
   - Add layer
   - Remove layer
   - Replace layer type
2. **Parameter-level mutations:**
   - Adjust channel dimensions
   - Modify kernel sizes
   - Change attention head count
3. **Topology mutations:**
   - Add skip connections
   - Remove connections
   - Modify connection pattern

### Selection Methods
1. **Tournament selection:** Best from random subset
2. **Pareto tournament:** Non-dominated from subset
3. **Novelty selection:** Encourage exploration

### Diversity Maintenance
1. **Crowding distance:** NSGA-II approach
2. **Novelty metric:** Distance to k-nearest neighbors
3. **Speciation:** Maintain sub-populations with different characteristics

## Implementation Recommendations

### Phase 1: Baseline Search Space
- Start with well-understood components (Conv, ResNet blocks, Self-Attention)
- Constrained parameter ranges (channels: 64-512, layers: 4-32)
- Sequential topology with optional skip connections
- Target: Stable search, reproducible results

### Phase 2: Advanced Components
- Cross-attention mechanisms
- Adaptive normalization (AdaGN, AdaLN)
- Conditional architectures (class, text conditioning)
- Target: Improved quality through novel components

### Phase 3: Hardware Optimization
- Apple Silicon profiling integration
- MPS kernel benchmarking
- Memory-aware fitness evaluation
- Target: Optimal deployment performance

### Phase 4: Multi-Objective Refinement
- NSGA-II/III for Pareto optimization
- Surrogate models for fast fitness estimation
- Active learning for expensive evaluations
- Target: Diverse set of high-quality architectures

## Metrics and Evaluation

### Quality Metrics
- **FID (Fréchet Inception Distance):** Standard for generative models
- **Inception Score (IS):** Diversity and quality measure
- **CLIP Score:** Text-image alignment for conditional generation
- **Human Evaluation:** Ultimate quality measure

### Efficiency Metrics
- **Inference Time:** Milliseconds per image
- **Memory Usage:** Peak GPU/unified memory
- **Energy Consumption:** Joules per image (battery impact)
- **Throughput:** Images per second

### Search Metrics
- **Convergence Rate:** Generations to reach plateau
- **Diversity Score:** Population heterogeneity
- **Pareto Front Quality:** Hypervolume indicator
- **Novelty:** Discovery of unique architectures

## Open Questions

1. **Search Space Design:**
   - How to balance search space size vs search efficiency?
   - Which architectural components are essential vs optional?

2. **Fitness Evaluation:**
   - Can we use surrogate models to reduce evaluation cost?
   - How to handle noisy fitness measurements?

3. **Hardware Optimization:**
   - Can we predict Apple Silicon performance without profiling?
   - How to optimize for multiple Apple Silicon variants (M1/M2/M3)?

4. **Generalization:**
   - Do architectures found on one dataset transfer to others?
   - Can we search on small datasets and scale to larger ones?

## References

1. Liu, H., et al. (2019). DARTS: Differentiable Architecture Search. ICLR.
2. Real, E., et al. (2019). Regularized Evolution for Image Classifier Architecture Search. AAAI.
3. Lu, Z., et al. (2019). NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm. GECCO.
4. Deb, K., et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Trans.
5. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
6. Cai, H., et al. (2019). ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. ICLR.

## Next Steps

1. ✅ Complete literature review
2. → Implement baseline evolutionary algorithm (EVOL-003)
3. → Define constrained search space (EVOL-004)
4. → Implement multi-objective optimization (EVOL-004)
5. → Hardware profiling integration (EVOL-005)
6. → Comprehensive benchmarking (EVOL-007)

---

**Last Updated:** 2025-10-17
**Review Status:** Initial draft - requires periodic updates as implementation progresses
