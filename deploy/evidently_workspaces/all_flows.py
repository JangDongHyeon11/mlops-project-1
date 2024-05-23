from detect_drift import detect_drift_flow


# for local testing
if __name__ == "__main__":
    detect_drift_flow('animals10_classifier_50px_trial7.yaml', last_days=5, last_n=50,
                evidently_project_name='production_model_monitor_dev')