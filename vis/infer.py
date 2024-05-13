from utils import *

if __name__ == "__main__":
    runs_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps"
    run_paths = {
        # Morph Random Baselines
        # "39(m)": "04-30-01-41-morph-sim-run-random-baselines/robust-sweep-1-id-21hzaq8h",
        "903(m)": "04-30-01-59-morph-sim-run-random-baselines/devoted-sweep-2-id-lipfnja5",
        "1847(m)": "04-30-01-59-morph-sim-run-random-baselines/fearless-sweep-3-id-r3ckwe3y",
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
        "903(j)": "04-30-02-22-morph-sim-run-random-baselines/silvery-sweep-19-id-8dtixb3h",
        "1847(j)": "04-30-02-22-morph-sim-run-random-baselines/summer-sweep-20-id-4skmh10v",
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
        "903(t=128)": "05-06-06-24-morph-sim-903-high-temp/tough-sweep-2-id-2ao10i82",
        "903(t=64)": "05-06-06-21-morph-sim-903-high-temp/solar-sweep-1-id-7mlasa01",
        "903(t=32)": "04-30-03-06-morph-sim-run-with-dataset/stoic-sweep-2-id-4kd1wvak",
        "903(t=1)": "04-30-05-48-morph-sim-run-lower-temp/driven-sweep-2-id-721585y2",
        # "8868(t=32)": "04-30-04-08-morph-sim-run-with-dataset/cerulean-sweep-22-id-f05uzgkb",

        # Joint Target
        "903(n=1, t=128)": "05-04-06-26-morph-sim-run-guided-high-temp/playful-sweep-9-id-c3fwbn7a",
        "903(n=0, t=128)": "05-04-06-11-morph-sim-run-guided-high-temp/ancient-sweep-3-id-9zds3qph",
        "1847(n=32,t=128)": "05-02-04-48-morph-sim-run-guided-high-temp/hearty-sweep-43-id-9hlis9sz",
        "1847(n=16,t=128)": "05-02-04-36-morph-sim-run-guided-high-temp/playful-sweep-11-id-bs4515ny",
        "1847(n=1,t=128)": "05-04-06-26-morph-sim-run-guided-high-temp/elated-sweep-10-id-wcekczbg",
        "1847(n=0,t=128)": "05-04-06-17-morph-sim-run-guided-high-temp/amber-sweep-4-id-5qt2b4ga",
    }

    # Load models and ground truth data
    assay_dataset = load_assay_matrix_from_csv()
    assay_model = load_assay_pred_model().to(device)
    cluster_labels = load_cluster_labels_from_csv()
    cluster_model = load_cluster_pred_model(return_ckpt=True)

    for run_name, run_id in run_paths.items():
        run_path = os.path.join(runs_dir, run_id)
        fps, rewards, smis = load_datum_from_run(runs_dir, run_id, remove_duplicates=False)
        assay_preds = predict_assay_logits_from_smi(run_path, smis, assay_model, torch.tensor([2]))
        cluster_preds = predict_cluster_logits_from_smi(run_path, smis, cluster_model, 0, force_recompute=True, use_gneprop=True)
