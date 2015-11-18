from PIL import Image
from scipy import misc

import numpy as np

def fileLister(path):
	import glob2
	files = glob2.glob(path)
	return sorted(files)


def read_resize_grayscale_invert_image(image):

	#Unlike MNIST image not resized to 20x20 (In MNIST resizing and normalization is done with antialiasing), also unlike MNIST it is not then centered in a 28x28 image using centre of mass or bounding box.
	#Rather it is directly resized to 28x8 
	#Can try the other approach later to see the effects of the different changes like centering method.

	#read an image resize it and convert it to grayscale
	im = Image.open(image).resize((28,28), Image.ANTIALIAS).convert('L')

	#creata numpy array out of the PIL image object created above
	im = misc.fromimage(im)


	#invert the above image
	im = np.invert(im)
	
	return im

def dense_to_one_hot(labels_dense, num_classes= 62):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot


def get_file_list(path1, path2, number_of_files = 'all'):
	print(number_of_files)
	dataset_list = fileLister(path1)

	dataset_size = 0
	for i in range(len(dataset_list)):
		with open(dataset_list[i],'r') as f:
			dataset_size += int(f.readline()[:-1])

	#print "Data size is :", dataset_size
	if number_of_files == 'all':
		list_of_filenames = fileLister(path2)
	else:
		list_of_filenames = fileLister(path2)
		list_of_filenames = list_of_filenames[:number_of_files]
		print("here to came")	
	print(len(list_of_filenames))

	return list_of_filenames
	

def extract_labels(dir1, dir2, number_of_images = 'all'):
	
	print(number_of_images)
	
	list_of_filenames = get_file_list(dir1, dir2, number_of_images)

	#print list_of_files[:1000]
	labels = []

	for file_name in list_of_filenames:
		labels.append
		val =  np.uint8(ord(file_name.split('_')[-2]))
		if val <=57:
			val = np.uint8(val - 48)
		elif 65 <= val <= 90:
			val = np.uint8(val - 65 + 10)
		elif  97 <= val <= 122:
			val = np.uint8(val - 96  + 36)
		labels.append(val)

	labels = np.array(labels, dtype= np.uint8)
	
	return labels






def extract_images(dir1, dir2, number_of_images = 'all'):

	list_of_filenames = get_file_list(dir1,dir2, number_of_images)
	data = []
	for file_name in list_of_filenames:
		im = read_resize_grayscale_invert_image(file_name)
		data.append(im)
	
	data = np.array(data)
	return data
	



class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # assert images.shape[3] == 1
      
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]




def read_data_sets(train_dir, fake_data = False, one_hot = False, number_of_images = 'all'):
	class DataSets(object):
		pass
	data_sets = DataSets()
	
	print(number_of_images)

	if fake_data:
		data_sets.train = DataSet([], [], fake_data=True)
		data_sets.validation = DataSet([], [], fake_data=True)
		data_sets.test = DataSet([], [], fake_data=True)
		return data_sets
	
	dir1 = '../character_dataset/sd_nineteen/**/D*.CLS'
	dir2 = '../character_dataset/sd_nineteen/HSF_4/**/*_*_*_*_D*_*_*_*_*.bmp'

	if number_of_images == all:
		TRAINSIZE = 40000
		VALIDATION_SIZE = 5000
	else:
		TRAINSIZE = int(number_of_images * 0.63)
		VALIDATION_SIZE = int(TRAINSIZE * 0.2)
	
	labels = extract_labels(dir1, dir2, number_of_images)
	train_labels = labels[:TRAINSIZE]
	test_labels = labels[TRAINSIZE:]
	print("labels extracted")	

	images = extract_images(dir1, dir2, number_of_images)
	train_images = images[:TRAINSIZE]
	test_images = images[TRAINSIZE:]
	print("images extracted")
	
	validation_images = train_images[:VALIDATION_SIZE]
	validation_labels = train_labels[:VALIDATION_SIZE]
	train_images = train_images[VALIDATION_SIZE:]
	train_labels = train_labels[VALIDATION_SIZE:]
	
	data_sets.train = DataSet(train_images, train_labels)
	data_sets.validation = DataSet(validation_images, validation_labels)
	data_sets.test = DataSet(test_images, test_labels)

	return data_sets



