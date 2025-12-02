# src/utils/balboni_plots.py
import matplotlib.pyplot as plt


def plot_wealth_distribution(panel):
    plt.figure(figsize=(8, 5))
    plt.hist(panel["wealth"].dropna(), bins=50)
    plt.title("Wealth Distribution (Qk_t)")
    plt.xlabel("Wealth")
    plt.ylabel("Frequency")
    plt.show()


def plot_health_distribution(panel):
    plt.figure(figsize=(8, 5))
    plt.hist(panel["health_index"].dropna(), bins=50)
    plt.title("Health Index Distribution")
    plt.xlabel("Health Index")
    plt.ylabel("Frequency")
    plt.show()


def plot_investment_distribution(panel):
    plt.figure(figsize=(8, 5))
    plt.hist(panel["investment_amount"].dropna(), bins=50)
    plt.title("Investment Amount Distribution (ΔQpAssets)")
    plt.xlabel("Investment Amount")
    plt.ylabel("Frequency")
    plt.show()

def plot_total_gain_distribution(panel):
    plt.figure(figsize=(8, 5))
    plt.hist(panel["total_gain_from_baseline"].dropna(), bins=50)
    plt.title("Total Gain from Baseline Distribution")
    plt.xlabel("Total Gain from Baseline")
    plt.ylabel("Frequency")
    plt.show()
