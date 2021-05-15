// Libraries Required for the project
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, greatest, lit, struct}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.rand



object ToxicCommentClassifier {
  def main(args: Array[String]): Unit ={
    if (args.length !=2){
      System.out.println("Usage: ToxicCommentClassifier InputDirectory OutputDirectory ")
    }
    val inputFilePath = args(0)
    val outputFilePath = args(1)

    // Building/Creating Spark Session
    val spark = SparkSession
      .builder()
      .appName("Toxic Comment Classification")
      .getOrCreate()
    import spark.implicits._

    // We have Preprocess the data in 3 steps
    //Step 1

    val inputFile1 = spark.
      read.option("header","true")
      .option("InferSchema","true")
      .csv(inputFilePath)

    val inputFile2 = inputFile1.filter($"comment_text".isNotNull).filter($"target".isNotNull).filter($"id" rlike "^[0-9]")

    val newData= inputFile2.na.fill("0.0")

    val preprocessData = newData.withColumn("target",when(col("target")<0.5,0).otherwise(1))

    // Step 2
    // Creating toxicity type label column

    val preprocessData2 = preprocessData.withColumn("NotApplicable",lit(0.000))
      .withColumn("severe_toxicity",col("severe_toxicity").cast("Double") )
      .withColumn("obscene",col("obscene").cast("Double"))
      .withColumn("identity_attack",col("identity_attack").cast("Double"))
      .withColumn("insult",col("insult").cast("Double"))
      .withColumn("threat",col("threat").cast("Double"))

    val structsType = preprocessData2.select("severe_toxicity","threat","obscene","insult",
      "identity_attack","NotApplicable").columns.map(
      s => struct(col(s).as("value"), lit(s).as("key"))
    )


    val preprocessData3 = preprocessData2.withColumn("ToxicityType", greatest(structsType: _*).getItem("key")).drop("severe_toxicity","threat","obscene","insult","identity_attack","NotApplicable")


    // Preprocessing Step 3
    // Here we have created labels for each category so that we can identify toxic comment on each topic


    val preprocessData4 = preprocessData3.drop("heterosexual", "homosexual_gay_or_lesbian", "hindu",
      "intellectual_or_learning_disability", "other_disability", "other_gender", "other_race_or_ethnicity",
      "other_religion", "other_sexual_orientation", "physical_disability", "psychiatric_or_mental_illness")

    val preprocessData5 = preprocessData4.withColumn("NotApplicable",lit(0.000))
      .withColumn("asian",col("asian").cast("Double") )
      .withColumn("buddhist",col("buddhist").cast("Double"))
      .withColumn("transgender",col("transgender").cast("Double"))
      .withColumn("bisexual",col("bisexual").cast("Double"))
      .withColumn("black",col("black").cast("Double"))
      .withColumn("male",col("male").cast("Double"))
      .withColumn("female",col("female").cast("Double"))
      .withColumn("jewish",col("jewish").cast("Double"))
      .withColumn("latino",col("latino").cast("Double"))
      .withColumn("white",col("white").cast("Double"))
      .withColumn("christian",col("christian").cast("Double") )
      .withColumn("muslim",col("muslim").cast("Double"))
      .withColumn("atheist",col("atheist").cast("Double"))


    val structsType2 = preprocessData5.select("asian","buddhist","transgender","bisexual","black","male","female",
      "jewish","latino","white","christian","muslim","atheist","NotApplicable")
      .columns.map(
      s => struct(col(s).as("value"), lit(s).as("key"))
    )


    val preprocessData6 = preprocessData5.withColumn("ToxicityTopic", greatest(structsType2: _*).getItem("key"))
      .drop("asian","buddhist","transgender","bisexual","black","male","female",
        "jewish","latino","white","christian","muslim","atheist","NotApplicable","publication_id","parent_id","article_id","rating","funny","sad","wow","likes","disagree","sexual_explicit","toxicity_annotator_count","identity_annotator_count","created_date")


    val affirmative = preprocessData6.filter("target=0")
    val assertive = preprocessData6.filter("target=1")
    val reducedAffirmative = affirmative.sample(false, 0.5)
    val DataSampled = assertive.union(reducedAffirmative)
    val finalData = DataSampled.orderBy(rand())


    // Model Training and Testing phase

    // The Model will be trained and tested by implementing two classifier
    // Logistic Regression and Random Forest Classifier

    val tk = new Tokenizer().setInputCol("comment_text").setOutputCol("words")

    val stpwd = new StopWordsRemover().setInputCol(tk.getOutputCol).setOutputCol("stpwd")



    val hTF = new HashingTF().setInputCol(stpwd.getOutputCol).setOutputCol("features")

    // Creating Labels
    val LabelTarget = new StringIndexer().setInputCol("target").setOutputCol("label")

    val ToxicLabelType = new StringIndexer().setInputCol("ToxicityType").setOutputCol("label1")

    val ToxicLabelTopic = new StringIndexer().setInputCol("ToxicityTopic").setOutputCol("label2")

    // Logistic Regression Model

    val modelLR = new LogisticRegression().setMaxIter(10).setFeaturesCol(hTF.getOutputCol).setLabelCol(LabelTarget.getOutputCol)

    val modelLR2 = new LogisticRegression().setMaxIter(10).setFeaturesCol(hTF.getOutputCol).setLabelCol(ToxicLabelType.getOutputCol)

    val modelLR3 = new LogisticRegression().setMaxIter(10).setFeaturesCol("features").setLabelCol(ToxicLabelTopic.getOutputCol)

    // Random Forest Classifier

    val modelRF = new RandomForestClassifier().setLabelCol(LabelTarget.getOutputCol).setFeaturesCol(hTF.getOutputCol).setNumTrees(10)

    val modelRF2 = new RandomForestClassifier().setLabelCol(ToxicLabelType.getOutputCol).setFeaturesCol(hTF.getOutputCol).setNumTrees(10)

    val modelRF3 = new RandomForestClassifier().setLabelCol(ToxicLabelTopic.getOutputCol).setFeaturesCol(hTF.getOutputCol).setNumTrees(10)


    // Creating/Building LR model pipeline
    val pipelineLR1 = new Pipeline().setStages(Array(tk, stpwd, hTF, LabelTarget, modelLR))
    val GridparamLR = new ParamGridBuilder().addGrid(hTF.numFeatures, Array( 10, 100, 1000)).addGrid(modelLR.regParam, Array(0.1,0.01)).build()

    val pipelineLR2 = new Pipeline().setStages(Array(tk, stpwd, hTF, ToxicLabelType, modelLR2))
    val GridparamLR2 = new ParamGridBuilder().addGrid(hTF.numFeatures, Array(10, 100, 1000)).addGrid(modelLR2.regParam, Array(0.1,0.01)).build()

    val pipelineLR3 = new Pipeline().setStages(Array(tk, stpwd, hTF, ToxicLabelTopic, modelLR3))
    val GridparamLR3 = new ParamGridBuilder().addGrid(hTF.numFeatures, Array(10, 100, 1000)).addGrid(modelLR3.regParam, Array(0.1, 0.01)).build()


    // Creating/Building RF model pipeline

    val pipelineRF = new Pipeline().setStages(Array(tk, stpwd, hTF, LabelTarget, modelRF))
    val GridparamRF = new ParamGridBuilder().addGrid(hTF.numFeatures, Array(10, 100, 1000)).build()

    val pipelineRF2 = new Pipeline().setStages(Array(tk, stpwd, hTF, ToxicLabelType, modelRF2))
    val GridparamRF2 = new ParamGridBuilder().addGrid(hTF.numFeatures, Array(10, 100, 1000)).build()

    val pipelineRF3 = new Pipeline().setStages(Array(tk, stpwd, hTF, ToxicLabelTopic, modelRF3))
    val GridparamRF3 = new ParamGridBuilder().addGrid(hTF.numFeatures, Array(10, 100, 1000)).build()


    // Evaluating our Models :

    // Cross Validation and Pipeline for LR Model
    val evaluatorLR = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val cvLR = new CrossValidator().setEstimator(pipelineLR1).setEvaluator(evaluatorLR).setEstimatorParamMaps(GridparamLR).setNumFolds(3)


    val evaluatorLR2 = new MulticlassClassificationEvaluator().setLabelCol("label1").setPredictionCol("prediction")
    val cvLR2 = new CrossValidator().setEstimator(pipelineLR2).setEvaluator(evaluatorLR2).setEstimatorParamMaps(GridparamLR2).setNumFolds(3)


    val evaluatorLR3 = new MulticlassClassificationEvaluator().setLabelCol("label2").setPredictionCol("prediction")
    val cvLR3 = new CrossValidator().setEstimator(pipelineLR3).setEvaluator(evaluatorLR3).setEstimatorParamMaps(GridparamLR3).setNumFolds(3)


    // Cross Validation and Pipeline for RF Model
    val evaluatorRF = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val cvRF = new CrossValidator().setEstimator(pipelineRF).setEvaluator(evaluatorRF).setEstimatorParamMaps(GridparamRF).setNumFolds(3)


    val evaluatorRF2 = new MulticlassClassificationEvaluator().setLabelCol("label1").setPredictionCol("prediction")
    val cvRF2 = new CrossValidator().setEstimator(pipelineRF2).setEvaluator(evaluatorRF2).setEstimatorParamMaps(GridparamRF2).setNumFolds(3)


    val evaluatorRF3 = new MulticlassClassificationEvaluator().setLabelCol("label2").setPredictionCol("prediction")
    val cvRF3 = new CrossValidator().setEstimator(pipelineRF3).setEvaluator(evaluatorRF3).setEstimatorParamMaps(GridparamRF3).setNumFolds(3)


    // Model Training for LR Model 1 where label == "Target"


    val Array(trainingLR, testLR) = finalData.randomSplit(Array(0.8,0.2))

    val cvModelLR = cvLR.fit(trainingLR)
    val transforModelLR = cvModelLR.transform(testLR)

    // Model Training for LR Model 2 where label == "ToxicType"

    val Array(trainingLR2, testLR2) = finalData.randomSplit(Array(0.8,0.2))
    val cvModelLR2 = cvLR2.fit(trainingLR2)
    val transforModelLR2 = cvModelLR2.transform(testLR2)

    // Model Training for LR Model 3 where label == "ToxicTopic"

    val Array(trainingLR3, testLR3) = finalData.randomSplit(Array(0.8,0.2))
    val cvModelLR3 = cvLR3.fit(trainingLR3)
    val transforModelLR3 = cvModelLR3.transform(testLR3)


    // Model Training for RF Model 1 where label == "Target"

    val Array(trainingRF, testRF) = finalData.randomSplit(Array(0.8,0.2))

    val cvModelRF = cvRF.fit(trainingRF)
    val transforModelRF = cvModelRF.transform(testRF)

    // Model Training for RF Model 2 where label == "ToxicType"

    val Array(trainingRF2, testRF2) = finalData.randomSplit(Array(0.8,0.2))
    val cvModelRF2 = cvRF2.fit(trainingRF2)
    val transforModelRF2 = cvModelRF2.transform(testRF2)


    // Model Training for RF Model 3 where label == "ToxicTopic"


    val Array(trainingRF3, testRF3) = finalData.randomSplit(Array(0.8,0.2))
    val cvModelRF3 = cvRF3.fit(trainingRF3)
    val transforModelRF3 = cvModelRF3.transform(testRF3)



    // Output and Accuracy for the Model
    // Prediction and Evaluation for LR Model 1 when "Target" is label

    cvModelLR.bestModel.params.toString
    var output =""

    val PredictionAndLabelsLR = transforModelLR.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics = new MulticlassMetrics(PredictionAndLabelsLR)

    val confusionMatrix1 = metrics.confusionMatrix
    output += "Confusion Matrix for LR Model 1: \n" + confusionMatrix1+ "\n"

    val accuracyLR =metrics.accuracy
    output += "Accuracy for LR Model 1: \n" + accuracyLR + "\n"

    val precisionLR = metrics.weightedPrecision
    output += "Precision for LR Model 1: \n" + precisionLR + "\n"

    val recallLR = metrics.weightedRecall
    output += "Recall for LR Model 1: \n" + recallLR + "\n"

    val fmeasureLR = metrics.weightedFMeasure
    output += "F1Score for LR Model 1: \n" + fmeasureLR + "\n"

    output+="\n"



    // Prediction and Evaluation for LR Model 2 when "ToxicType" is label


    cvModelLR2.bestModel.params.toString

    val PredictionAndLabelsLR2 = transforModelLR2.select("prediction", "label1").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics2 = new MulticlassMetrics(PredictionAndLabelsLR2)

    val confusionMatrix2 = metrics2.confusionMatrix
    output += "Confusion Matrix for LR Model 2: \n" + confusionMatrix2 + "\n"

    val accuracyLR2 =metrics2.accuracy
    output += "Accuracy for LR Model 2: \n" + accuracyLR2 + "\n"

    val precisionLR2 = metrics2.weightedPrecision
    output += "Precision for LR Model 2: \n" + precisionLR2 + "\n"

    val recallLR2 = metrics2.weightedRecall
    output += "Recall for LR Model 2: \n" + recallLR2 + "\n"

    val fmeasureLR2 = metrics2.weightedFMeasure
    output += "F1Score for LR Model 2: \n" + fmeasureLR2 + "\n"

    output+="\n"


    // Prediction and Evaluation for LR Model 3 when "ToxicTopic" is label

    cvModelLR3.bestModel.params.toString

    val PredictionAndLabelsLR3 = transforModelLR3.select("prediction", "label2").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics3 = new MulticlassMetrics(PredictionAndLabelsLR3)

    val confusionMatrix3 = metrics3.confusionMatrix
    output += "Confusion Matrix for LR Model 3: \n" + confusionMatrix3 + "\n"

    val accuracyLR3 =metrics3.accuracy
    output += "Accuracy for LR Model 3: \n" + accuracyLR3 + "\n"

    val precisionLR3 = metrics3.weightedPrecision
    output += "Precision for LR Model 3: \n" + precisionLR3 + "\n"

    val recallLR3 = metrics3.weightedRecall
    output += "Recall for LR Model 3: \n" + recallLR3 + "\n"

    val fmeasureLR3 = metrics3.weightedFMeasure
    output += "F1Score for LR Model 3: \n" + fmeasureLR3 + "\n"

    output+="\n"



    // Prediction and Evaluation for RF Model 1 when "Target" is label

    cvModelRF.bestModel.params.toString

    val PredictionAndLabelsRF = transforModelRF.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics4 = new MulticlassMetrics(PredictionAndLabelsRF)

    val confusionMatrix4 = metrics4.confusionMatrix
    output += "Confusion Matrix for RF Model 1: \n" + confusionMatrix4 + "\n"

    val accuracyRF =metrics4.accuracy
    output += "Accuracy for RF Model 1: \n" + accuracyRF + "\n"

    val precisionRF = metrics4.weightedPrecision
    output += "Precision for RF Model 1: \n" + precisionRF + "\n"

    val recallRF = metrics4.weightedRecall
    output += "Recall for RF Model 1: \n" + recallRF + "\n"

    val fmeasureRF = metrics4.weightedFMeasure
    output += "F1Score for RF Model 1: \n" + fmeasureRF + "\n"

    output+="\n"

    // Prediction and Evaluation for RF Model 2 when "ToxicType" is label


    cvModelRF2.bestModel.params.toString

    val PredictionAndLabelsRF2 = transforModelRF2.select("prediction", "label1").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics5 = new MulticlassMetrics(PredictionAndLabelsRF2)

    val confusionMatrix5 = metrics4.confusionMatrix
    output += "Confusion Matrix for RF Model 2: \n" + confusionMatrix5 + "\n"

    val accuracyRF2 =metrics5.accuracy
    output += "Accuracy for RF Model 2: \n" + accuracyRF2 + "\n"

    val precisionRF2 = metrics5.weightedPrecision
    output += "Precision for RF Model 2: \n" + precisionRF2 + "\n"

    val recallRF2 = metrics5.weightedRecall
    output += "Recall for RF Model 2: \n" + recallRF2 + "\n"

    val fmeasureRF2 = metrics5.weightedFMeasure
    output += "F1Score for RF Model 2: \n" + fmeasureRF2 + "\n"

    output+="\n"


    // Prediction and Evaluation for RF Model 3 when "ToxicTopic" is label

    cvModelRF3.bestModel.params.toString

    val PredictionAndLabelsRF3 = transforModelRF3.select("prediction", "label2").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    val metrics6 = new MulticlassMetrics(PredictionAndLabelsRF3)

    val confusionMatrix6 = metrics6.confusionMatrix
    output += "Confusion Matrix for RF Model 3: \n" + confusionMatrix6 + "\n"

    val accuracyRF3 =metrics6.accuracy
    output += "Accuracy for RF Model 3: \n" + accuracyRF3 + "\n"

    val precisionRF3 = metrics6.weightedPrecision
    output += "Precision for RF Model 3: \n" + precisionRF3 + "\n"

    val recallRF3 = metrics6.weightedRecall
    output += "Recall for RF Model 3: \n" + recallRF3 + "\n"

    val fmeasureRF3 = metrics6.weightedFMeasure
    output += "F1Score for RF Model 3: \n" + fmeasureRF3 + "\n"

    output+="\n"



    val sc = spark.sparkContext
    val outputRdd: RDD[String] = sc.parallelize(List(output));
    outputRdd.coalesce(1, true).saveAsTextFile(outputFilePath)


  }

}
