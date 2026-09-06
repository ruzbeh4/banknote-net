import subprocess
import sys
import re
import os
import matplotlib.pyplot as plt

# ==========================================
# SETTINGS
# ==========================================
# Change this to the exact name of the script you provided above
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


TARGET_SCRIPT = "./src/predict_custom.py"

# Generate thresholds from 50 to 100 in steps of 5 (0.50 to 1.00)
thresholds = [i / 100.0 for i in range(55, 100, 5)]

final_accuracies = []
banknote_accuracies = []


def main():
    print(f"Starting threshold evaluation loop on {TARGET_SCRIPT}...\n")

    for threshold in thresholds:
        print(f"Running test for threshold: {threshold:.2f}...", end=" ", flush=True)

        # Build the command using the current Python executable
        cmd = [sys.executable, TARGET_SCRIPT, "--threshold", str(threshold)]



        # Execute the script and capture its console output
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        output = result.stdout

        # If there's an error in the sub-script, print it and stop
        if result.returncode != 0:
            print("FAILED!")
            print(f"Error output:\n{result.stderr}")
            sys.exit(1)

        # Parse the percentages from the output using Regular Expressions
        acc_match = re.search(r"Final Accuracy:.*?\(([\d\.]+)%\)", output)
        banknote_match = re.search(r"Accuracy without background noise:.*?\(([\d\.]+)%\)", output)

        if acc_match and banknote_match:
            acc = float(acc_match.group(1))
            banknote_acc = float(banknote_match.group(1))

            final_accuracies.append(acc)
            banknote_accuracies.append(banknote_acc)

            print(f"Overall: {acc:.2f}% | Banknote: {banknote_acc:.2f}%")
        else:
            print("ERROR parsing output!")
            print(f"Script output was:\n{output}")
            sys.exit(1)

    # ==========================================
    # PLOTTING
    # ==========================================
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))

    # Plot both lines
    plt.plot(thresholds, final_accuracies, marker='o', linestyle='-', color='blue', label='Overall Accuracy')
    plt.plot(thresholds, banknote_accuracies, marker='s', linestyle='--', color='green',
             label='Banknote Accuracy (No Noise)')

    # Formatting the plot
    plt.title('Model Accuracy vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy (%)')

    # Format X-axis to show exactly the steps we tested
    plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds])
    plt.yticks(range(0, 105, 10))  # Y-axis from 0 to 100

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left')

    # Save the plot as an image file and show it
    plt.savefig('threshold_evaluation_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'threshold_evaluation_plot.png'.")

    plt.show()


if __name__ == "__main__":
    main()