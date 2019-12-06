import os, numpy as np, sys, cv2
from PIL import Image
from utils import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts, random_colors
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration
import json
max_color = 30
colors = random_colors(max_color)       # Generate random colors
type_whitelist = [0, 1, 2, 3, 4, 5, 6, 7]


def show_image_with_boxes(img, objects_res, objects_res_special, object_gt, calib, save_path,  img_height, img_width):
	img2 = np.copy(img)

	for obj in objects_res:
		color_tmp = tuple([int(tmp * 255) for tmp in colors[obj['track_id'] % max_color]])
		text = 'ID: %d  ' % (obj['track_id'])
		if obj['appear'] == 0:
			cv2.rectangle(img2, (int(obj['box2d'][0]), int(obj['box2d'][1])), (int(obj['box2d'][2]), int(obj['box2d'][3])), color_tmp, 2)
		else:
			cv2.rectangle(img2, (int(obj['box2d'][0]), int(obj['box2d'][1])), (int(obj['box2d'][2]), int(obj['box2d'][3])), (0,255,0), 6)
			cv2.line(img2, (int(obj['box2d'][0]), int(obj['box2d'][1])),
						  (int(obj['box2d'][2]), int(obj['box2d'][3])), (0, 255, 0), 6)
			cv2.line(img2, (int(obj['box2d'][2]), int(obj['box2d'][1])),
						  (int(obj['box2d'][0]), int(obj['box2d'][3])), (0, 255, 0), 6)
		img2 = cv2.putText(img2, text, (int(0.5 * (obj['box2d'][0] + obj['box2d'][2])), int(obj['box2d'][1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp)

	for obj in objects_res_special:
		# print('disappear embed ---------')
		# from IPython import embed
		# embed()
		color_tmp = tuple([int(tmp * 255) for tmp in colors[obj['track_id'] % max_color]])
		cv2.rectangle(img2, (int(obj['box2d'][0]), int(obj['box2d'][1])), (int(obj['box2d'][2]), int(obj['box2d'][3])), (0,0,255), 6)
		cv2.line(img2, (int(obj['box2d'][0]), int(obj['box2d'][1])), (int(obj['box2d'][2]), int(obj['box2d'][3])),
					  (0, 0, 255), 6)
		cv2.line(img2, (int(obj['box2d'][2]), int(obj['box2d'][1])), (int(obj['box2d'][0]), int(obj['box2d'][3])),
					  (0, 0, 255), 6)
		text = 'ID: %d  ' % (obj['track_id'])
		img2 = cv2.putText(img2, text, (int(0.5 * (obj['box2d'][0] + obj['box2d'][2])), int(obj['box2d'][1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp)


	img = Image.fromarray(img2)
	img = img.resize((img_width, img_height))
	img.save(save_path)

def vis(data_root, result_root):

	with open(os.path.join(data_root, 'kitti_train.json'), 'r', encoding='utf-8') as f:
		data = json.load(f)

	video_names = []
	for data_info_tmp in data:
		if data_info_tmp['video_id'] not in video_names:
			video_names.append(data_info_tmp['video_id'])

	# num_images = len(data)
	# 21 videos

	for this_video_name in video_names:
		this_video = []
		for tmp_frame in data:
			if tmp_frame['video_id'] == this_video_name:
				this_video.append(tmp_frame)
		num_images = len(this_video)
		print('number of images to visualize is %d' % num_images)
		start_count = 0

		for count in range(start_count, num_images):
			this_img_info = this_video[count]
			image_path_tmp = data_root + '/' + 'image_02' + '/' + this_img_info['filename']
			image_index = int(fileparts(image_path_tmp)[1])
			image_tmp = np.array(Image.open(image_path_tmp))
			img_height, img_width, img_channel = image_tmp.shape
			vis_dir = os.path.join(result_root, 'vis/%s' % this_video_name)
			save_3d_bbox_dir = os.path.join(vis_dir)
			mkdir_if_missing(vis_dir)
			# result_tmp = os.path.join(result_dir)		# load the result

			# if not is_path_exists(result_tmp): object_res = []
			# else: object_res = read_label(result_tmp)
			# print('processing index: %d, %d/%d, results from %s' % (image_index, count+1, num_images, result_tmp))
			appear = 0
			ID_pool = []
			object_res_filtered = []
			object_special_former = []
			for tmp_index, labels_index in enumerate(this_img_info['ann']['labels']):
				if labels_index not in type_whitelist: continue
				# if hasattr(this_img_info, 'score'):
				# 	if this_img_info.score < score_threshold: continue
				ID_pool.append(this_img_info['ann']['track_id'][tmp_index])

				if image_index != 0:
					if this_img_info['ann']['track_id'][tmp_index] not in ID_pool_former:
						appear = 1

				buf = {'box2d': this_img_info['ann']['bboxes'][tmp_index],
					   'track_id': this_img_info['ann']['track_id'][tmp_index],
					   'appear': appear}
				appear = 0
				object_res_filtered.append(buf)

			if image_index != 0:
				for obj_tmp in object_res_filtered_former:
					if obj_tmp['track_id'] not in ID_pool:
						object_special_former.append(obj_tmp)

			num_instances = len(object_res_filtered)
			save_image_with_3dbbox_gt_path = os.path.join(save_3d_bbox_dir, '%06d.jpg' % (image_index))
			show_image_with_boxes(image_tmp, object_res_filtered, object_special_former, [], [], save_image_with_3dbbox_gt_path, img_height, img_width)
			object_res_filtered_former = object_res_filtered.copy()
			ID_pool_former = ID_pool.copy()
			print('number of objects to plot is %d' % (num_instances))
			count += 1

if __name__ == "__main__":
	if len(sys.argv)!=2:
		print("Usage: python visualization.py result_sha(e.g., car_3d_det_test)")
		sys.exit(1)
	result_root = './results'
	result_sha = sys.argv[1]
	if 'val' in result_sha: data_root = '/backdata01/datasets/kitti/tracking/training'
	elif 'test' in result_sha: data_root = '/backdata01/datasets/kitti/tracking/testing'
	else:
		print("wrong split!")
		sys.exit(1)
	vis(data_root, result_root)