from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os, csv


class customizedEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, sentences1, sentences2, scores):

        super(customizedEmbeddingSimilarityEvaluator, self).__init__(sentences1, sentences2, scores)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        embeddings1 = model.encode(
            self.sentences1,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings2 = model.encode(
            self.sentences2,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        # logger.info(
        #     "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_cosine, eval_spearman_cosine)
        # )
        # logger.info(
        #     "Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #         eval_pearson_manhattan, eval_spearman_manhattan
        #     )
        # )
        # logger.info(
        #     "Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #         eval_pearson_euclidean, eval_spearman_euclidean
        #     )
        # )
        # logger.info(
        #     "Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_dot, eval_spearman_dot)
        # )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        eval_pearson_cosine,
                        eval_spearman_cosine,
                        eval_pearson_euclidean,
                        eval_spearman_euclidean,
                        eval_pearson_manhattan,
                        eval_spearman_manhattan,
                        eval_pearson_dot,
                        eval_spearman_dot,
                    ]
                )

            return np.mean([eval_pearson_cosine, eval_pearson_euclidean, eval_pearson_manhattan, eval_pearson_dot,
                            eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot])