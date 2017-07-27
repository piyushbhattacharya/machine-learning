import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics


Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header", "true").option("inferSchema","true").format("csv").load("titanic.csv")

val logregdataall = data.select(data("Survived").as("label"), $"Pclass", $"Name", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked")

val logregdata = logregdataall.na.drop()

//data.printSchema

// Transform string to numeric values
val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("sexIndex")
val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

// convert numeric values to Binary 1 and 0
val genderEncoder = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVector")
val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVector")

// (label, features)
val assembler = new VectorAssembler().setInputCols(Array("Pclass", "sexVector", "Age", "SibSp", "Parch", "Fare", "EmbarkedVector")).setOutputCol("features")

val Array(training, test) = logregdata.randomSplit(Array(0.7,0.3), seed = 12345)

val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkedIndexer, genderEncoder, embarkedEncoder, assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)

// display the predicted values
results.select($"label", $"prediction").show()

val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion Matrix ")
println(metrics.confusionMatrix)
