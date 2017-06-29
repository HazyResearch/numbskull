
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

import org.apache.log4j.Logger
import org.apache.log4j.Level


object Example1 {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder().appName("Example1").config("spark.master", "local").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val voterDF = spark.read.format("csv").option("header", "true").load("voter.csv")

    val yesVotersDF = voterDF.filter("vote = \"yes\"")
    val noVotersDF = voterDF.filter("vote = \"no\"")

    val issuesDF = voterDF.select("issue_id").distinct()

    voterDF.show()
    yesVotersDF.show()
    noVotersDF.show()
    issuesDF.show()

    val yesVoters = FGTemplate.VTempl(yesVotersDF, 2)
    val noVoters = FGTemplate.VTempl(noVotersDF, 2)
    val issues = FGTemplate.VTempl(issuesDF, 2)

    val voteWeight = FGTemplate.WTempl(Seq())

    val yesFactors = FGTemplate.FTemplAnd(Seq(issues, yesVoters), voteWeight)
    val noFactors = FGTemplate.FTemplAnd(Seq(issues.negate, noVoters), voteWeight)

    val fgt = FGTemplate(yesFactors, noFactors)

    println(fgt)

    println(fgt.varTemplates)

    spark.stop()
  }
}
