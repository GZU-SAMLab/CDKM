import itertools
import json
import os

from third_party.CenterNet2.detectron2.evaluation import COCOEvaluator
from third_party.CenterNet2.detectron2.structures import BoxMode
from third_party.CenterNet2.detectron2.utils.file_io import PathManager


class CDKMVGEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            assert input["image_id"] == int(input['file_name'].split('/')[-1].split('.')[0])
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"], output_logits=True)
                h = input['height']
                w = input['width']
                scale = 720.0 / max(h, w)
                scaled_inst = []
                for inst in prediction["instances"]:
                    inst['bbox'][0] = inst['bbox'][0] * scale
                    inst['bbox'][1] = inst['bbox'][1] * scale
                    inst['bbox'][2] = inst['bbox'][2] * scale
                    inst['bbox'][3] = inst['bbox'][3] * scale
                    scaled_inst.append(inst)
                if len(scaled_inst) > 0:
                    prediction["instances"] = scaled_inst
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        '''
        This is only for saving the results to json file
        '''
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "vg_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()


def instances_to_coco_json(instances, img_id, output_logits=False):
    """
        Add object_descriptions and logit (if applicable) to
        detectron2's instances_to_coco_json
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    object_descriptions = instances.pred_object_descriptions.data
    if output_logits:
        logits = instances.logits.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            'object_descriptions': object_descriptions[k],
        }
        if output_logits:
            result["logit"] = logits[k]

        results.append(result)
    return results