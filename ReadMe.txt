Toxic Comment Classification

--------------------- Files Included -------------------------------------------------------------------------------------------------------------
README.txt
Report
Output.txt
-------------------------------------------------------------------------------------------------------------------------------------------------

Steps to create "toxic-comment-classification_2.11-0.1.jar" jar file:

1. Go to root directory of the project (Toxic Comment Classification).
2. Go to sbt shell and type following command
	sbt:Toxic Comment Classification> package

The above command will generate jar under target/scala-2.11/ directory by the name of toxic-comment-classification_2.11-0.1.jar.

--------------------- Steps to run on AWS ----------------------------------------------------------------------------------------------------------------------------------------------
Data: Data can be downloaded from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
Copy train.csv to AWS S3 and note the path to the File.
Run the file on the AWS EMR cluster by: 
	- Add Steps
	- Step type: Select "Spark Application"
	- Name:Toxic Comment Classification
	- JAR location: Select the jar file: s3://Bucket_Name/toxic-comment-classification_2.11-0.1.jar
	- Enter in spark-submit options:--class "ToxicCommentClassifier"
	
	- Arguments: s3://Bucket_Name/train.csv s3://Bucket_Name/output (arguments separated by space)
	- Output files will be generated in output folder on S3

Sample output is included in the Toxic Comment Classification Folder based on the output.
