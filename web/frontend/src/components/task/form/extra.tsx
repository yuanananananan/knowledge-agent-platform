import { taskFormOpts, withForm } from '@/hooks/task/form';
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldTitle,
} from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';

export const ExtraField = withForm({
  ...taskFormOpts,
  render: ({ form }) => (
    <Card className="rounded border-none">
      <CardHeader>
        <CardTitle>参数配置</CardTitle>
        <CardDescription>针对当前模态的任务，配置详细训练参数</CardDescription>
      </CardHeader>

      <CardContent>
        <form.AppField name="extra" mode="array">
          {(field) => (
            <FieldGroup className="space-y-5">
              {field.state.value.map((item, i) => {
                const currentType = item.type;

                return (
                  <Field key={i} className="lg:w-2/3 xl:w-1/2">
                    {currentType === 'boolean' ? (
                      <div className="flex items-center justify-between rounded-lg border border-border/60 px-4 py-3">
                        <div className="pr-4">
                          <FieldTitle className="text-base">{item.name}</FieldTitle>
                          <FieldDescription className="mt-1 text-sm text-muted-foreground">
                            {item.description}
                          </FieldDescription>
                        </div>

                        <form.AppField name={`extra[${i}].value`}>
                          {(subField) => (
                            <Switch
                              checked={Boolean(subField.state.value)}
                              onCheckedChange={(checked) => {
                                subField.handleChange(checked);
                              }}
                            />
                          )}
                        </form.AppField>
                      </div>
                    ) : (
                      <>
                        <FieldContent>
                          <FieldTitle>{item.name}</FieldTitle>
                          <FieldDescription>{item.description}</FieldDescription>
                        </FieldContent>

                        <form.AppField name={`extra[${i}].value`}>
                          {(subField) => (
                            <Input
                              id={`extra_${i}_value`}
                              type="text"
                              inputMode={currentType === 'number' ? 'decimal' : undefined}
                              placeholder={currentType === 'number' ? '请输入数值，例如 0.3' : ''}
                              value={
                                subField.state.value === null || subField.state.value === undefined
                                  ? ''
                                  : String(subField.state.value)
                              }
                              onChange={(e) => {
                                const raw = e.target.value;

                                if (currentType === 'number') {
                                  // 不在输入过程中强转 Number，避免 0.3 / 0.1 输入被打断
                                  subField.handleChange(raw);
                                } else {
                                  subField.handleChange(raw);
                                }
                              }}
                            />
                          )}
                        </form.AppField>
                      </>
                    )}
                  </Field>
                );
              })}
            </FieldGroup>
          )}
        </form.AppField>
      </CardContent>
    </Card>
  ),
});