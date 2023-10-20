import os
import datasets
from datasets.tasks import AutomaticSpeechRecognition

class MevoiceVoiceConfig(datasets.BuilderConfig):
    def __init__(self, name, sub_version, **kwargs):
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.validated_hr_total = kwargs.pop("val_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        self.num_of_voice = kwargs.pop("num_of_voice", None)
        description = f"MeVoice dataset"
        super(MevoiceVoiceConfig, self).__init__(
            name=name, version=datasets.Version("1.0.0", ""), description=description, **kwargs
        )

class MevoiceVoice(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "phone": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
            task_templates=[
                AutomaticSpeechRecognition(audio_file_path_column="path", transcription_column="phone")
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        abs_path_to_data = os.path.join("./data/", self.config.name)
        abs_path_to_clips = os.path.join(abs_path_to_data, "clips") 
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": abs_path_to_data + "/test.tsv",
                    "path_to_clips": abs_path_to_clips,
                },
            )
        ]

    def _generate_examples(self, filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        path_idx = data_fields.index("path")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            
            print(lines[0])
            
            headline = lines[0]

            column_names = headline.strip().split("\t")
            
            assert (
                column_names == data_fields
            ), f"The file should have {data_fields} as column names, but has {column_names}"

            for id_, line in enumerate(lines[1:]):
                field_values = line.strip().split("\t")
                field_values[path_idx] = os.path.join(path_to_clips, field_values[path_idx])
                if len(field_values) < len(data_fields):
                    field_values += (len(data_fields) - len(field_values)) * ["''"]

                yield id_, {key: value for key, value in zip(data_fields, field_values)}
