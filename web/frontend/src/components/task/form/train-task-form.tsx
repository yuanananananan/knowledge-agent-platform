import { taskFormOpts, useAppForm } from '@/hooks/task/form';
import { ExtraField } from './extra';
import { BasicField } from './basic';
import { useExtraConfigs } from '@/hooks/task/use-extra-configs';
import { useDatasets } from '@/hooks/dataset/use-datasets';
import { useClassifiers } from '@/hooks/classifier/use-classifiers';
import { useBackbones } from '@/hooks/backbone/use-backbones';
import type { Modality } from '@/types/dataset';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../../ui/card';
import { RadioGroup, RadioGroupItem } from '../../ui/radio-group';
import { Field, FieldContent, FieldLabel, FieldTitle } from '../../ui/field';
import { useEffect, useState, useMemo } from 'react';
import { FeatureField } from './feature';
import { useFeatures } from '@/hooks/feature/use-features';
import { useFeatureGroups } from '@/hooks/feature/use-feature-groups';
import { useStartTraining } from '@/hooks/train';
import { useAuthStore } from '@/stores/auth';
import { useNavigate } from 'react-router';
import { getModalityOptions } from '@/lib/modality';
import type { ExtraConfigDTO } from '@/services/api/task';

interface TrainTaskFormProps {
  defaultModality?: Modality;
}

const hiddenExtraNames = [
  'adaptive',
  'is_nas',
  'is_GNN',
  'feature_combinations',
  'backbone',
];

function filterExtraConfigs(configs: ExtraConfigDTO[]) {
  return configs.filter((item) => !hiddenExtraNames.includes(item.name));
}

export default function TrainTaskForm({
  defaultModality = 'SAR',
}: TrainTaskFormProps) {
  const [modality, setModality] = useState<Modality>(defaultModality);

  const { data: extraConfigs = { RD: [], SAR: [], MON: [] } } = useExtraConfigs();
  const { data: datasets = [] } = useDatasets();
  const { data: classifiers = [] } = useClassifiers();
  const { data: backbones = [] } = useBackbones();
  const { data: features = [] } = useFeatures();
  const { data: featureGroups = [] } = useFeatureGroups();

  const navigate = useNavigate();
  const { mutate: startTraining } = useStartTraining();
  const user = useAuthStore((state) => state.user);

  const form = useAppForm({
    ...taskFormOpts,
    onSubmit: async ({ value }) => {
      const selectedBackbone = backbones.find(
        (item) => item.id === value.backboneId,
      );

      const isNas = selectedBackbone?.backboneType === 'nas_search';

      startTraining(
        {
          userId: user.id,
          datasetId: value.datasetId,
          nodes: {},
          backboneId: value.backboneId,
          classifierId: value.classifierId,

          // is_GNN=false 时，不传特征算子
          featureIds: value.isGNN ? value.featureIds : [],

          // 这两个才是现在真正需要传给后端的字段
          isGNN: value.isGNN,
          isNas,

          extra: value.extra.map((item) => ({
            ...item,
            value:
              item.type === 'number'
                ? Number(item.value)
                : item.type === 'boolean'
                  ? Boolean(item.value)
                  : String(item.value),
          })),
        },
        {
          onSuccess: () => {
            navigate('/task');
          },
        },
      );
    },
  });

  const filteredDatasets = useMemo(() => {
    return datasets.filter((dataset) => dataset.modality === modality);
  }, [datasets, modality]);

  const filteredFeatures = useMemo(() => {
    return features.filter((feature) => feature.modality === modality);
  }, [features, modality]);

  const filteredFeatureGroups = useMemo(() => {
    return featureGroups.filter(
      (group) =>
        group.features.length > 0 &&
        group.features[0].modality === modality,
    );
  }, [featureGroups, modality]);

  const filteredBackbones = useMemo(() => {
    return backbones.filter((backbone) => {
      if (!backbone.modality) {
        return true;
      }

      return backbone.modality.toUpperCase() === modality;
    });
  }, [backbones, modality]);

  useEffect(() => {
    form.setFieldValue('extra', filterExtraConfigs(extraConfigs[modality]));
  }, [modality, extraConfigs, form]);

  const handleModalityChange = (value: Modality) => {
    setModality(value);
    form.setFieldValue('extra', filterExtraConfigs(extraConfigs[value]));
    form.setFieldValue('datasetId', 0);
    form.setFieldValue('backboneId', 0);
    form.setFieldValue('featureIds', []);
  };

  return (
    <form
      className="space-y-2"
      onSubmit={(e) => {
        e.preventDefault();
        form.handleSubmit();
      }}
    >
      <Card className="rounded border-none">
        <CardHeader>
          <CardTitle>模态</CardTitle>
          <CardDescription>从 SAR、RD、MON 三种模态中选择一种</CardDescription>
        </CardHeader>

        <CardContent>
          <RadioGroup
            value={modality}
            onValueChange={handleModalityChange}
            className="flex w-full gap-3 lg:w-2/3 xl:w-1/2"
          >
            {getModalityOptions().map((option) => (
              <FieldLabel key={option.value} htmlFor={`modality-${option.value}`}>
                <Field orientation="horizontal">
                  <FieldContent>
                    <FieldTitle>{option.label}</FieldTitle>
                  </FieldContent>
                  <RadioGroupItem
                    value={option.value}
                    id={`modality-${option.value}`}
                  />
                </Field>
              </FieldLabel>
            ))}
          </RadioGroup>
        </CardContent>
      </Card>

      <BasicField
        form={form}
        datasetItems={filteredDatasets}
        classifierItems={classifiers}
        backboneItems={filteredBackbones}
      />

      <form.Subscribe selector={(state) => state.values.isGNN}>
        {(isGNN) =>
          isGNN ? (
            <FeatureField
              form={form}
              featureItems={filteredFeatures}
              featureGroups={filteredFeatureGroups}
            />
          ) : null
        }
      </form.Subscribe>

      <ExtraField form={form} />

      <form.AppForm>
        <form.SubmitButton />
      </form.AppForm>
    </form>
  );
}