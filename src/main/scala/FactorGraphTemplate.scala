
import org.apache.spark.sql.DataFrame



trait GenericVariableTemplateAlias[R] {
  def apply(args: String*): GenericVariableTemplateAlias[R]
  def map[S](f: R => S): GenericVariableTemplateAlias[S]
}

class VariableTemplate[T](df: DataFrame, domain: Seq[T]) {
  def asAlias: VariableTemplateAlias[T, T] = VariableTemplateAlias[T, T](this, df.columns, (x) => x)
  def arity: Int = df.columns.length
}

case class VariableTemplateAlias[T, R](vt: VariableTemplate[T], attrs: Seq[String], mapper: T => R) extends GenericVariableTemplateAlias[R] {
  assert(attrs.length == vt.arity)

  override def apply(args: String*): GenericVariableTemplateAlias[R] = VariableTemplateAlias[T, R](vt, args, mapper)

  override def map[S](f: R => S): GenericVariableTemplateAlias[S] = VariableTemplateAlias[T, S](vt, attrs, x => f(mapper(x)))
}


class WeightTemplate(ar: Int) {
  def arity: Int = ar
}

case class WeightTemplateAlias(wt: WeightTemplate, attrs: Seq[String]) {
  assert(attrs.length == wt.arity)

  def apply(args: String*): WeightTemplateAlias = WeightTemplateAlias(wt, args)
}


trait GenericFactorTemplate

case class FactorTemplate[R](terms: Seq[GenericVariableTemplateAlias[R]], weight: WeightTemplateAlias, reducer: (R, R) => R, energy: R => Double, filter: DataFrame => DataFrame) extends GenericFactorTemplate


