import matplotlib.pyplot as plt

# Values
labels = ['Sync Guardrail', 'IGUANA Swarm']
latencies = [69.09, 26.03]
colors = ['#4d4d4d', '#b3b3b3']  # Dark gray and light gray

# Create figure
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Bar plot
bars = ax.bar(labels, latencies, color=colors, edgecolor='black', width=0.4, linewidth=1.5)

# Styling axes
ax.set_ylabel('Latency per Token (ms)', fontsize=18, fontweight='normal', labelpad=10)
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))

# Grid lines (horizontal only, dashed, behind bars)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=(0, (5, 5)), color='gray', linewidth=0.8)

# Show ticks on all sides pointing inward
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=16)

# Set X-axis labels font size
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=18)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save as PDF
plt.savefig('_paper/ISSE_26_IGUANA/latency_plot.pdf', format='pdf', bbox_inches='tight')
print("Successfully generated latency_plot.pdf with values: 69.09 ms and 26.03 ms.")
