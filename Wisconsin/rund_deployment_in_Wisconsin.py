from utils.cluster.cluster import ClusterSubmitter

from processing.runs.run_database import RunDatabase

output_dir = "OUTPUT_DIRECTORY"

#
# Create submitter tool
#

with ClusterSubmitter(
    job_name="graphnet_deployment_2_percent",
    flush_factor=1,
    memory=4000, # 4 GB
    disk_space=1000, # 1 GB
    output_dir=os.path.join(output_dir, "cluster"),
    start_up_commands=["cd <your_folder>", "eval <your_CVMFS_setup>", "source <your_virtual_environment>"], 	#CHECK
    env_shell="<your_CVMFS_icetray",    									                                                    #CHECK
    run_locally=True, # Set to true to run locally for testing (instead of submitting to cluster) 						#CHECK
) as submitter :


    #
    # Loop over runs and files
    #

    # Create the run database
    run_db = RunDatabase(
        database_file=database_file,
        data_type="data",
        pass_num=2,																		#CHECK
        dataset="12", 	# This is the year
        subsample="test", # None is all data, burn is 10%, test is 2%
    )

    # Get good runs
    stage = "level2"
    run_ids = run_db.get_run_ids(stage=stage)
    for run_id in run_ids:

        # Get the GCD file for this run
        gcd_file = run_db.get_gcd_file(run_id)

        # Get the i3 files for this run
        i3_files, i3_files_metadata = run_db.get_i3_files(stage=stage, run_id=run_id, only_existing=True, split=False)

        # Loop over i3 files
        for i3_file in i3_files :

            #
            # Run command to process the i3 files
            #

            # Define the command to run
            #command = "python <your script path>" # Must be an absolute path								#CHECK
            command = "python <deployment_py_file" 				#CHECK
            command += f" -g {gcd_file}"												#CHECK
            command += f" -i {i3_file}"													#CHECK
            #TODO you can fill this out

            # Add it to the submission
            submitter.add(command=command)												#CHECK
