import subprocess

# Define your workflows
workflows = {
    "1": {
        "name": "Trade Day Workflow",
        "scripts": [
            "Trade Day Stats.py",
            "RTH Pivot Levels.py",
            "Opening Range.py",
            "ON Pivot Levels.py",
            "Afternoon Pivot Levels.py"
        ]
    },
    "2": {
        "name": "Daily Stats Workflow",
        "scripts": [
            "2HR Pivot Levels with Hits.py",
            "Pivot Levels - Hits.py",
            "Pivot Levels.py",
            "Pivot Levels - Weekly.py",
            "4HR Averages & Pivots.py",
            "Range Extensions.py"
        ]
    },
    "3": {
        "name": "Run All Workflows",
        "scripts": []  # Will be filled dynamically
    }
}

# Fill option 3 with all scripts from workflows 1 and 2
workflows["3"]["scripts"] = workflows["1"]["scripts"] + workflows["2"]["scripts"]

def run_scripts(scripts):
    for script in scripts:
        try:
            subprocess.run(["python3", script], check=True)
            print(f"✅ {script} completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script}: {e}")
            break
    else:
        print("\n🎉 Workflow finished successfully!")

if __name__ == "__main__":
    print("Select a workflow to run:")
    for key, wf in workflows.items():
        print(f"{key}: {wf['name']}")
    
    choice = input("\nEnter number: ").strip()
    
    if choice in workflows:
        print(f"\n🚀 Starting {workflows[choice]['name']}...\n")
        run_scripts(workflows[choice]["scripts"])
    else:
        print("❌ Invalid choice")
