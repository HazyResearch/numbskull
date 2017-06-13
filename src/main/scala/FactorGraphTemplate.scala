
import org.apache.spark.sql.DataFrame



class VariableTemplate(df: DataFrame, domain_arity: Int) {
  def asAlias: VariableTemplateAlias = VariableTemplateAlias(this, df.columns, (x) => x)
  def arity: Int = df.columns.length
}

case class VariableTemplateAlias(vt: VariableTemplate, attrs: Seq[String], mapper: Int => Any) {
  assert(attrs.length == vt.arity)

  def apply(args: String*) = VariableTemplateAlias(vt, args, mapper)

  def map(f: Any => Any) = VariableTemplateAlias(vt, attrs, x => f(mapper(x)))

  def negate = map(x => if (x == 0) 1 else 0)
}


class WeightTemplate(ar: Int) {
  def arity: Int = ar
}

case class WeightTemplateAlias(wt: WeightTemplate, attrs: Seq[String]) {
  assert(attrs.length == wt.arity)

  def apply(args: String*): WeightTemplateAlias = WeightTemplateAlias(wt, args)
}


case class FactorTemplate(terms: Seq[VariableTemplateAlias], weight: WeightTemplateAlias, reducer: (Any, Any) => Any, energy: Any => Double, filter: DataFrame => DataFrame) {
  def scale(s: Double): FactorTemplate = FactorTemplate(terms, weight, reducer, x => energy(x) * s, filter)
}


object FGTemplate {
  def VTempl(df: DataFrame, domain_arity: Int): VariableTemplateAlias = VariableTemplateAlias(new VariableTemplate(df, domain_arity), df.columns, x => x)
  def WTempl(attrs: Seq[String]): WeightTemplateAlias = WeightTemplateAlias(new WeightTemplate(attrs.length), attrs)
  
  def FTemplAnd(terms: Seq[VariableTemplateAlias], weight: WeightTemplateAlias): FactorTemplate = FactorTemplate(terms, weight, (x, y) => x.asInstanceOf[Boolean] && y.asInstanceOf[Boolean], x => if (x.asInstanceOf[Boolean]) 1.0 else -1.0, df => df)
  def FTemplOr(terms: Seq[VariableTemplateAlias], weight: WeightTemplateAlias): FactorTemplate = FactorTemplate(terms, weight, (x, y) => x.asInstanceOf[Boolean] || y.asInstanceOf[Boolean], x => if (x.asInstanceOf[Boolean]) 1.0 else -1.0, df => df)
  
  def FTemplRatio(terms: Seq[VariableTemplateAlias], weight: WeightTemplateAlias): FactorTemplate = FactorTemplate(
    terms.map(t => t.map(x => if (x.asInstanceOf[Boolean]) 1 else 0)), 
    weight, 
    (x, y) => x.asInstanceOf[Int] + y.asInstanceOf[Int], 
    x => Math.log(1.0 + x.asInstanceOf[Int]), 
    df => df)
}

