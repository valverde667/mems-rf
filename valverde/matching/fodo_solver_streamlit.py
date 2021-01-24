"""Script to investigate parameter settings on solving the KV equation. The
current MEMs lattice cell is asymmetrical and it is not obvious what causes the
difficulty in getting suitable focusing. The purpose of this script is to
incrementally tweak the FODO cell and solve the KV equations until the MEMS
MEMS cell is created."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pdb

import warp

import parameters
from solver import hard_edge_kappa, solve_KV
