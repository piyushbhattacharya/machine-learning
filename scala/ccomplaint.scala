/**
* Consumer complaints classification using Logistic Regression
**/

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DateType, TimestampType};
import org.apache.spark.sql.functions._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()

// Define custom Schema
val customSchema = StructType(Array(
       StructField("Date received", DateType, true), //dropped
       StructField("Product", StringType, true), //dummified
       StructField("Sub-product", StringType, true), //dummified
       StructField("Issue", StringType, true), //dummified
       StructField("Sub-issue", StringType, true), //dummified
       StructField("Consumer complaint narrative", StringType, true), //dummified
       StructField("Company public response", StringType, true), //dummified
       StructField("Company", StringType, true), //dropped
       StructField("State", StringType, true), //dummified
       StructField("ZIP code", IntegerType, true), //dropped
       StructField("Tags", StringType, true), //dummified
       StructField("Consumer consent provided?", StringType, true),//dummified
       StructField("Submitted via", StringType, true), //dummified
       StructField("Date sent to company", DateType, true), //dropped
       StructField("Company response to consumer", StringType, true), //dummified
       StructField("Timely response?", StringType, true), //dummified
       StructField("Consumer disputed?", StringType, true), //to numeric
       StructField("Complaint ID", IntegerType, true) //dropped
   ))

   val data = spark.read.option("header", "true").format("csv").option("dateFormat", "MM/dd/yyyy").option("inferSchema", "false").option("treatEmptyValuesAsNulls", "true").schema(customSchema).load("Consumer_Complaints_train.csv")

//Prepare the data
  val datediff_df = data.withColumn("date_diff", datediff($"Date sent to company", $"Date received"))

  //drop the date columns
  val dsdf = data.drop($"Date received").drop($"Date sent to company").drop($"Complaint ID").drop($"ZIP code").drop($"Consumer complaint narrative").drop($"Company")

  val str2num = udf((cval: String) => {
    cval match {
      case c if (c == "Yes") => 1
      case c if (c == "No") => 0
      case _ => 0
    }
  })

  val label_df = dsdf.withColumn("Consumer disputed?", str2num($"Consumer disputed?"))

  //label_df.printSchema

  val logregdataall = (label_df.select(label_df("Consumer disputed?").as("label"), $"Product", $"Sub-product", $"Issue",
                                  $"Sub-issue", $"Company public response", $"State", $"Tags",
                                  $"Consumer consent provided?", $"Submitted via", $"Company response to consumer", $"Timely response?"))

  val logregdata = logregdataall.na.drop()

// 1. Product
val productIndexer = new StringIndexer().setInputCol("Product").setOutputCol("productIndex").setHandleInvalid("skip")
val productEncoder = new OneHotEncoder().setInputCol("productIndex").setOutputCol("productVector")

// 2. Sub product
val sproductIndexer = new StringIndexer().setInputCol("Sub-product").setOutputCol("sproductIndex").setHandleInvalid("skip")
val sproductEncoder = new OneHotEncoder().setInputCol("sproductIndex").setOutputCol("sproductVector")

//3. Issue
val issueIndexer = new StringIndexer().setInputCol("Issue").setOutputCol("issueIndex").setHandleInvalid("skip")
val issueEncoder = new OneHotEncoder().setInputCol("issueIndex").setOutputCol("issueVector")

//4. Sub-issue
val sissueIndexer = new StringIndexer().setInputCol("Sub-issue").setOutputCol("sissueIndex").setHandleInvalid("skip")
val sissueEncoder = new OneHotEncoder().setInputCol("sissueIndex").setOutputCol("sissueVector")

//5. Consumer complaint narrative
//val ccnIndexer = new StringIndexer().setInputCol("Consumer complaint narrative").setOutputCol("ccnIndex")
//val ccnEncoder = new OneHotEncoder().setInputCol("ccnIndex").setOutputCol("ccnVector")

//5. Company public response
val cpubrespIndexer = new StringIndexer().setInputCol("Company public response").setOutputCol("cpubrespIndex").setHandleInvalid("skip")
val cpubrespEncoder = new OneHotEncoder().setInputCol("cpubrespIndex").setOutputCol("cpubrespVector")

//6. State
val stateIndexer = new StringIndexer().setInputCol("State").setOutputCol("stateIndex").setHandleInvalid("skip")
val stateEncoder = new OneHotEncoder().setInputCol("stateIndex").setOutputCol("stateVector")

//7. Tags
val tagsIndexer = new StringIndexer().setInputCol("Tags").setOutputCol("tagsIndex").setHandleInvalid("skip")
val tagsEncoder = new OneHotEncoder().setInputCol("tagsIndex").setOutputCol("tagsVector")

//8. Consumer consent provided?
val cconsIndexer = new StringIndexer().setInputCol("Consumer consent provided?").setOutputCol("cconsIndex").setHandleInvalid("skip")
val cconsEncoder = new OneHotEncoder().setInputCol("cconsIndex").setOutputCol("cconsVector")

//9. Submitted via
val submittedIndexer = new StringIndexer().setInputCol("Submitted via").setOutputCol("submittedIndex").setHandleInvalid("skip")
val submittedEncoder = new OneHotEncoder().setInputCol("submittedIndex").setOutputCol("submittedVector")

//10. Company response to consumer
val crespIndexer = new StringIndexer().setInputCol("Company response to consumer").setOutputCol("crespIndex").setHandleInvalid("skip")
val crespEncoder = new OneHotEncoder().setInputCol("crespIndex").setOutputCol("crespVector")

//11. Timely response?
val trespIndexer = new StringIndexer().setInputCol("Timely response?").setOutputCol("trespIndex").setHandleInvalid("skip")
val trespEncoder = new OneHotEncoder().setInputCol("trespIndex").setOutputCol("trespVector")

val assembler = (new VectorAssembler().setInputCols(Array("productVector", "sproductVector", "issueVector", "sissueVector", "cpubrespVector",
                "stateVector", "tagsVector", "cconsVector", "submittedVector", "crespVector", "trespVector")).setOutputCol("features"))

logregdata.cache
val Array(training, test) = logregdata.randomSplit(Array(0.7,0.3), seed = 12345)

println("Training split cols ......................... ")
val trcols = training.columns

println("Test split cols ......................... ")
val tcols = test.columns

val lr = new LogisticRegression()

val pipeline = (new Pipeline().setStages(Array(productIndexer, sproductIndexer, issueIndexer, sissueIndexer, cpubrespIndexer, stateIndexer, tagsIndexer, cconsIndexer, submittedIndexer, crespIndexer, trespIndexer,
                               productEncoder, sproductEncoder, issueEncoder, sissueEncoder, cpubrespEncoder, stateEncoder, tagsEncoder, cconsEncoder, submittedEncoder, crespEncoder, trespEncoder, assembler, lr)))

val model = pipeline.fit(training)
println("Model is fitted to the pipeline .................................... ")

val results = model.transform(test)
println("Model is applied to test data  ...................................... ")

val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Metrics have been calculated  ...................................... ")
// display the head rows
println("Confusion Matrix ...................................................")
println(metrics.confusionMatrix)
//println(metrics.accuracy)
println(s"Accuracy: ${metrics.accuracy} Recall: ${metrics.recall}")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}
