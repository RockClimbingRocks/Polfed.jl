using DrWatson; 
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

# include("../../src/src.jl")
# using HDF5, Statistics, PythonPlot, LaTeXStrings, UnPack, DrWatson, CurveFit, DataFrames, DataInterpolations, LsqFit, Polynomials, StatsBase, PythonCall, Printf

using HDF5
using LaTeXStrings
using PythonPlot
using CSV
using DataFrames
# const matplotlib = PythonPlot.matplotlib
# const pyplot = PythonPlot.pyplot

# using QSystem.ChaosIndicators
# using Distributions, Roots
pyplot.style.use(["rok-custom"])
plotting_colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
matplotlib.rcParams["figure.dpi"] = 1000 
