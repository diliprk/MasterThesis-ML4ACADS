# MasterThesis - Machine Learning for an Aircraft's Cabin Air Distribution System
This repository contains all the implementations of my Master Thesis work at Airbus Operations GmbH.
 - **Hybrid Physics Informed Neural Network (Novel proposed architecture)** - The most notable achievement in my master thesis is the development of this novel Hybrid - Physics Informed Neural Network (`PINN`) architecture to optimize the time and resource-intensive physical simulation workflows in the ATA21-21 customization process of the A350 aircraft. The Hybrid PINN works in tandem with FDDN (Airbus proprietary 1D fluid flow solver) to predict correction factors for FDDN to get optimal flow rates.
 - **CycleGANs** - were used to generate synthetic data to address the sparse training data from the lab test results and improve the restrictor predictions of the cabin air distribution system.
 - **Bayesian Neural Networks** - Bayesian Neural Network architectures (`MC Dropout` and `Bayesian Ensembling`) were implemented for uncertainty measures in the restrictor predictions of the cabin air distribution system.

All implementations are placed in their respective folders
