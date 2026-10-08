import { createFormHookContexts, createFormHook, formOptions } from '@tanstack/react-form';
import SubmitButton from '../../components/task/form/submit-button';
import type { ExtraConfigDTO } from '@/services/api/task';
import SelectField from '@/components/task/form/selectField';

export const { fieldContext, useFieldContext, formContext, useFormContext } =
  createFormHookContexts();

export const { useAppForm, withForm } = createFormHook({
  fieldContext,
  formContext,
  fieldComponents: { SelectField },
  formComponents: {
    SubmitButton,
  },
});

export const taskFormOpts = formOptions({
  defaultValues: {
    datasetId: 0,
    backboneId: 0,
    classifierId: 0,
    featureIds: [] as number[],
    isGNN: true,
    isNas: false,
    extra: [] as ExtraConfigDTO[],
  },
});
