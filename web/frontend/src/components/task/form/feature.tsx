import { taskFormOpts, withForm } from '@/hooks/task/form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { Feature, FeatureGroupWithDetails } from '@/types/feature';
import { Field, FieldLabel } from '@/components/ui/field';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckSquare, Square, MinusSquare } from 'lucide-react';

export const FeatureField = withForm({
  ...taskFormOpts,
  props: {
    featureItems: [] as Feature[],
    featureGroups: [] as FeatureGroupWithDetails[],
  },
  render: ({ form, featureItems, featureGroups }) => (
    <Card className="rounded border-none">
      <CardHeader>
        <CardTitle>特征配置</CardTitle>
        <CardDescription>选择要加入网络中的特征算子或特征组合</CardDescription>
      </CardHeader>
      <CardContent>
        <form.AppField name="featureIds">
          {(field) => (
            <Tabs defaultValue="features" className="w-full">
              <TabsList>
                <TabsTrigger value="features" className="gap-2">
                  单个特征
                  {featureItems.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {featureItems.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="groups" className="gap-2">
                  特征组合
                  {featureGroups.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {featureGroups.length}
                    </Badge>
                  )}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="features" className="mt-4">
                {featureItems.length === 0 ? (
                  <div className="flex items-center justify-center h-48 text-muted-foreground">
                    <p>当前模态下暂无可用特征</p>
                  </div>
                ) : (
                  <ScrollArea className="h-96 lg:w-2/3 xl:w-1/2 pr-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {featureItems.map((item) => (
                        <div
                          key={item.id}
                          className="group rounded-lg border p-3 transition-colors hover:bg-accent hover:border-accent-foreground/20"
                        >
                          <Field orientation="horizontal" className="gap-3">
                            <Checkbox
                              id={`feature-checkbox-${item.id}`}
                              name={field.name}
                              checked={field.state.value.includes(item.id)}
                              onCheckedChange={(checked) => {
                                if (checked) {
                                  field.pushValue(item.id);
                                } else {
                                  const index = field.state.value.indexOf(item.id);
                                  if (index > -1) {
                                    field.removeValue(index);
                                  }
                                }
                              }}
                            />
                            <FieldLabel
                              htmlFor={`feature-checkbox-${item.id}`}
                              className="flex-1 font-normal cursor-pointer"
                            >
                              <div className="flex flex-col gap-1">
                                <span className="text-sm font-medium">{item.name}</span>
                                {item.description && (
                                  <span className="text-xs text-muted-foreground line-clamp-2">
                                    {item.description}
                                  </span>
                                )}
                              </div>
                            </FieldLabel>
                          </Field>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>

              <TabsContent value="groups" className="mt-4">
                {featureGroups.length === 0 ? (
                  <div className="flex items-center justify-center h-48 text-muted-foreground">
                    <p>当前模态下暂无可用特征组合</p>
                  </div>
                ) : (
                  <ScrollArea className="h-96 lg:w-2/3 xl:w-1/2 pr-4">
                    <div className="space-y-3">
                      {featureGroups.map((group) => {
                        const allFeaturesSelected = group.features.every((feature) =>
                          field.state.value.includes(feature.id)
                        );
                        const someSelected =
                          !allFeaturesSelected &&
                          group.features.some((feature) => field.state.value.includes(feature.id));

                        return (
                          <div
                            key={group.id}
                            className="rounded-lg border bg-card transition-all hover:shadow-md"
                          >
                            <div className="p-4 space-y-3">
                              <div className="flex items-start justify-between gap-3">
                                <div className="flex-1 space-y-1">
                                  <div className="flex items-center gap-2">
                                    <h4 className="font-semibold text-sm">{group.name}</h4>
                                    <Badge variant="secondary" className="text-xs">
                                      {group.featureCount} 个特征
                                    </Badge>
                                  </div>
                                  {group.description && (
                                    <p className="text-sm text-muted-foreground">
                                      {group.description}
                                    </p>
                                  )}
                                </div>
                                <Button
                                  type="button"
                                  variant="secondary"
                                  size="sm"
                                  className="shrink-0 gap-1.5"
                                  onClick={() => {
                                    if (allFeaturesSelected) {
                                      // 取消选择所有该组的特征
                                      group.features.forEach((feature) => {
                                        const index = field.state.value.indexOf(feature.id);
                                        if (index > -1) {
                                          field.removeValue(index);
                                        }
                                      });
                                    } else {
                                      // 选择所有该组的特征
                                      group.features.forEach((feature) => {
                                        if (!field.state.value.includes(feature.id)) {
                                          field.pushValue(feature.id);
                                        }
                                      });
                                    }
                                  }}
                                >
                                  {allFeaturesSelected ? (
                                    <>
                                      <CheckSquare className="h-4 w-4" />
                                      取消全选
                                    </>
                                  ) : someSelected ? (
                                    <>
                                      <MinusSquare className="h-4 w-4" />
                                      补全选择
                                    </>
                                  ) : (
                                    <>
                                      <Square className="h-4 w-4" />
                                      全选
                                    </>
                                  )}
                                </Button>
                              </div>

                              <div className="rounded-md bg-muted/50 p-2">
                                <p className="text-xs text-muted-foreground font-medium mb-1.5">
                                  包含特征:
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                  {group.features.map((f) => (
                                    <Badge
                                      key={f.id}
                                      variant={
                                        field.state.value.includes(f.id) ? 'default' : 'outline'
                                      }
                                      className="text-xs"
                                    >
                                      {f.name}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>
            </Tabs>
          )}
        </form.AppField>
      </CardContent>
    </Card>
  ),
});
