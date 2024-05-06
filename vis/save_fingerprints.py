from batch_plot import *
from rdkit.Chem import rdFingerprintGenerator

def load_datum_from_run(run_dir, run_id, remove_duplicates=True, save_fps=True):
    run_path = os.path.join(run_dir, run_id)
    if not os.path.exists(run_path):
        print(f"Run {run_id} does not exist")
        return None
    values = sqlite_load(f"{run_path}/train/", sqlite_cols, 1)
    smis, rewards = np.array(values['smi'][0]), np.array(values['fr_0'][0])
    original_len = len(smis)
    fps_file = os.path.join(run_path, "fps.npy")

    if remove_duplicates:
        smis, idx = np.unique(smis, return_index=True)
        rewards = rewards[idx]
        print(f"Removed {original_len - len(smis)} duplicates")

    if os.path.exists(fps_file):
        fps = np.load(fps_file)
        fps = list(map(get_fp_from_base64, fps))
        if len(fps) == original_len and remove_duplicates:
            fps = [fps[i] for i in idx]
        assert len(fps) == len(smis), f"fps len {len(fps)} != smis len {len(smis)}"
        print(f"Loaded fps from {fps_file}!")
    else:
        print("Generating fps...")
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
        fps = fpgen.GetFingerprints(mols, numThreads=8)
        if save_fps:
            print("Saving fps to file...")
            to_save_fps = np.array([x.ToBase64() for x in tqdm(fps)])
            np.save(fps_file, to_save_fps)
            print(f"Saved fps to {fps_file}")
    
    return fps, rewards, smis


if __name__ == "__main__":
    runs_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps"
    run_paths = {
        # # Morph Random Baselines
        # "39(m)": "04-30-01-41-morph-sim-run-random-baselines/robust-sweep-1-id-21hzaq8h",
        # "903(m)": "04-30-01-59-morph-sim-run-random-baselines/devoted-sweep-2-id-lipfnja5",
        # "1847(m)": "04-30-01-59-morph-sim-run-random-baselines/fearless-sweep-3-id-r3ckwe3y",
        # "2288(m)": "04-30-02-01-morph-sim-run-random-baselines/likely-sweep-4-id-ypcvjb7u",
        # "6888(m)": "04-30-02-01-morph-sim-run-random-baselines/hardy-sweep-5-id-aleb8abd",
        # "8838(m)": "04-30-02-01-morph-sim-run-random-baselines/jolly-sweep-6-id-cx14juh1",
        # "10075(m)": "04-30-02-01-morph-sim-run-random-baselines/jumping-sweep-7-id-i8wcg6pm",
        # "13905(m)": "04-30-02-01-morph-sim-run-random-baselines/skilled-sweep-8-id-s6a7y4qg",
        # "4331(m)": "04-30-02-01-morph-sim-run-random-baselines/sandy-sweep-9-id-tccku4hz",
        # "8206(m)": "04-30-02-01-morph-sim-run-random-baselines/unique-sweep-10-id-jkfvu24j",
        # "338(m)": "04-30-02-01-morph-sim-run-random-baselines/eternal-sweep-11-id-4b0tyvig",
        # "8949(m)": "04-30-02-01-morph-sim-run-random-baselines/rare-sweep-12-id-tjpfzaom",
        # "9277(m)": "04-30-02-02-morph-sim-run-random-baselines/divine-sweep-13-id-xr16ullm",
        # "9300(m)": "04-30-02-07-morph-sim-run-random-baselines/fearless-sweep-14-id-7qn32tnf",
        # "9445(m)": "04-30-02-07-morph-sim-run-random-baselines/tough-sweep-15-id-bppa91l0",
        # "9476(m)": "04-30-02-21-morph-sim-run-random-baselines/silver-sweep-16-id-inmd5k39",
        # "12071(m)": "04-30-02-22-morph-sim-run-random-baselines/dry-sweep-17-id-6q1sa0cc",

        # # Joint Random Baselines
        # "39(j)": "04-30-02-22-morph-sim-run-random-baselines/lilac-sweep-18-id-imhlzwas",
        # "903(j)": "04-30-02-22-morph-sim-run-random-baselines/silvery-sweep-19-id-8dtixb3h",
        # "1847(j)": "04-30-02-22-morph-sim-run-random-baselines/summer-sweep-20-id-4skmh10v",
        # "2288(j)": "",
        # "6888(j)": "",
        # "8838(j)": "",
        # "10075(j)": "",
        # "13905(j)": "",
        # "4331(j)": "",
        # "8206(j)": "",
        # "338(j)": "",
        # "8949(j)": "",
        # "9277(j)": "",
        # "9300(j)": "",
        # "9445(j)": "",
        # "9476(j)": "",
        # "12071(j)": "",

        # Morph Only Target
        # "903(t=32)": "04-30-03-06-morph-sim-run-with-dataset/stoic-sweep-2-id-4kd1wvak",
        "903(t=1)": "04-30-05-48-morph-sim-run-lower-temp/driven-sweep-2-id-721585y2",
        # "8868(t=32)": "04-30-04-08-morph-sim-run-with-dataset/cerulean-sweep-22-id-f05uzgkb"

        # Joint Target
        # "1847(n=32,t=128)": "05-02-04-48-morph-sim-run-guided-high-temp/hearty-sweep-43-id-9hlis9sz",
        # "1847(n=16,t=128)": "05-02-04-36-morph-sim-run-guided-high-temp/playful-sweep-11-id-bs4515ny",
        # "1847(n=1,t=32)": "04-30-04-14-morph-sim-run-with-dataset/frosty-sweep-27-id-4fdd0n63",
        # "1847(n=0,t=32)": "04-30-03-24-morph-sim-run-with-dataset/sandy-sweep-11-id-njg67az5",
    }

    runs_datum = {}
    for run_name, run_id in run_paths.items():
        run_path = os.path.join(runs_dir, run_id)
        fps, rewards, smis = load_datum_from_run(runs_dir, run_id, remove_duplicates=False)
