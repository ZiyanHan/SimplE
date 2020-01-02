#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *

class SimplE_ignr(TensorFactorizer):

	def __init__(self, params, dataset="wn18"):
		TensorFactorizer.__init__(self, model_name="SimplE_ignr", loss_function="likelihood", params=params, dataset=dataset)

	def setup_weights(self):
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		self.rel_emb      = tf.get_variable(name="rel_emb",      initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.rel_inv_emb  = tf.get_variable(name="rel_inv_emb",  initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_head_emb = tf.get_variable(name="ent_head_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_tail_emb = tf.get_variable(name="ent_tail_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.var_list = [self.rel_emb, self.rel_inv_emb, self.ent_head_emb, self.ent_tail_emb]

	def define_regularization(self):
		self.regularizer = 2.0 * (tf.nn.l2_loss(self.ent_head_emb) + tf.nn.l2_loss(self.ent_tail_emb) + tf.nn.l2_loss(self.rel_emb) + tf.nn.l2_loss(self.rel_inv_emb)) / self.num_batch

	def gather_train_embeddings(self):
		self.h_emb = tf.concat( [tf.gather(self.ent_head_emb, self.head), tf.gather(self.ent_head_emb, self.tail)], 0 )
		self.t_emb = tf.concat( [tf.gather(self.ent_tail_emb, self.tail), tf.gather(self.ent_tail_emb, self.head)], 0 )
		self.r_emb = tf.concat( [tf.gather(self.rel_emb, self.rel), tf.gather(self.rel_inv_emb, self.rel)], 0 )

	def gather_test_embeddings(self):
		self.h_emb = tf.gather(self.ent_head_emb, self.head)
		self.t_emb = tf.gather(self.ent_tail_emb, self.tail)
		self.r_emb = tf.gather(self.rel_emb, self.rel)

	def create_train_model(self):
		self.init_scores = tf.reduce_sum(tf.multiply(tf.multiply(self.h_emb, self.r_emb), self.t_emb), 1)
		self.scores = tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.
		self.labels = tf.tile(self.y, [2])

	def create_test_model(self):
		self.init_scores = tf.reduce_sum(tf.multiply(tf.multiply(self.h_emb, self.r_emb), self.t_emb), 1)
		self.dissims = -tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.


	def predict_v2(self):
		import pandas as pd
		data = np.loadtxt("%s/test.txt" % self.dataset, delimiter="\t", dtype=np.str)
		head = []
		rel = []
		tail = []
		for line in data:
			for i in range(self.params.edge_type_num):
				head.append(self.reader.ent2id[line[0]])
				rel.append(i)
				tail.append(self.reader.ent2id[line[2]])
		pred = self.sess.run(self.dissims, feed_dict={self.head: head, self.rel: rel, self.tail: tail})
		pred = np.reshape(pred, (data.shape[0], self.params.edge_type_num))
		df = pd.DataFrame(pred)
		cols = []
		id2rel = dict((v,k) for k,v in self.reader.rel2id.items())
		id2en = dict((v,k) for k,v in self.reader.ent2id.items())
		for i in range(self.params.edge_type_num):
			cols.append(id2rel[i])
		df.columns = cols
		df.insert(0, "tail", [id2en[id] for id in tail[::self.params.edge_type_num]])
		df.insert(0, "head", [id2en[id] for id in head[::self.params.edge_type_num]])
		df.to_csv("%s/predictions_v2.csv" % self.dataset, index=False)
