import unittest
from typing import List

from vulcan.types.feature_types import (
    ActionContext,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    FeatureType,
    MCTSAction,
)


class TestFeatureDefinition(unittest.TestCase):
    def test_feature_definition_creation(self):
        fd = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.CODE_BASED,
            description="A test feature.",
            code="result = df['col1'] + df['col2']",
            dependencies=["col1", "col2"],
        )
        self.assertEqual(fd.name, "test_feature")
        self.assertEqual(fd.feature_type, FeatureType.CODE_BASED)
        self.assertTrue(fd.validate_definition())

    def test_code_based_validation(self):
        fd_valid = FeatureDefinition(
            name="f1", feature_type=FeatureType.CODE_BASED, description="d", code="c"
        )
        self.assertTrue(fd_valid.validate_definition())
        fd_invalid = FeatureDefinition(
            name="f2", feature_type=FeatureType.CODE_BASED, description="d"
        )
        self.assertFalse(fd_invalid.validate_definition())

    def test_llm_based_validation(self):
        fd_valid = FeatureDefinition(
            name="f1",
            feature_type=FeatureType.LLM_BASED,
            description="d",
            llm_prompt="p",
            text_columns=["t"],
        )
        self.assertTrue(fd_valid.validate_definition())
        fd_invalid_prompt = FeatureDefinition(
            name="f2",
            feature_type=FeatureType.LLM_BASED,
            description="d",
            text_columns=["t"],
        )
        self.assertFalse(fd_invalid_prompt.validate_definition())
        fd_invalid_cols = FeatureDefinition(
            name="f3",
            feature_type=FeatureType.LLM_BASED,
            description="d",
            llm_prompt="p",
        )
        self.assertFalse(fd_invalid_cols.validate_definition())

    def test_hybrid_validation(self):
        fd_valid_pre = FeatureDefinition(
            name="f1",
            feature_type=FeatureType.HYBRID,
            description="d",
            llm_prompt="p",
            text_columns=["t"],
            preprocessing_code="pre",
        )
        self.assertTrue(fd_valid_pre.validate_definition())
        fd_valid_post = FeatureDefinition(
            name="f2",
            feature_type=FeatureType.HYBRID,
            description="d",
            llm_prompt="p",
            text_columns=["t"],
            postprocessing_code="post",
        )
        self.assertTrue(fd_valid_post.validate_definition())
        fd_invalid = FeatureDefinition(
            name="f3",
            feature_type=FeatureType.HYBRID,
            description="d",
            llm_prompt="p",
            text_columns=["t"],
        )
        self.assertFalse(fd_invalid.validate_definition())


class TestFeatureSet(unittest.TestCase):
    def setUp(self):
        self.fd1 = FeatureDefinition(
            name="feat1",
            feature_type=FeatureType.CODE_BASED,
            description="d1",
            code="c1",
            computational_cost=1.0,
        )
        self.fd2 = FeatureDefinition(
            name="feat2",
            feature_type=FeatureType.CODE_BASED,
            description="d2",
            code="c2",
            computational_cost=2.0,
        )
        self.feature_set = FeatureSet(
            features=[self.fd1, self.fd2], action_taken=MCTSAction.ADD
        )

    def test_feature_set_creation(self):
        self.assertEqual(len(self.feature_set.features), 2)
        self.assertEqual(self.feature_set.action_taken, MCTSAction.ADD)

    def test_get_feature_by_name(self):
        self.assertEqual(self.feature_set.get_feature_by_name("feat1"), self.fd1)
        self.assertIsNone(self.feature_set.get_feature_by_name("non_existent_feat"))

    def test_get_total_cost(self):
        self.assertEqual(self.feature_set.get_total_cost(), 3.0)
        empty_set = FeatureSet(features=[], action_taken=MCTSAction.ADD)
        self.assertEqual(empty_set.get_total_cost(), 0.0)


class TestActionContext(unittest.TestCase):
    def setUp(self):
        self.fd1 = FeatureDefinition(
            name="f1",
            feature_type=FeatureType.CODE_BASED,
            description="d1",
            code="c1",
            computational_cost=10,
        )
        self.fd2 = FeatureDefinition(
            name="f2",
            feature_type=FeatureType.CODE_BASED,
            description="d2",
            code="c2",
            computational_cost=20,
        )
        self.features_list = [self.fd1, self.fd2]
        self.feature_set = FeatureSet(
            features=self.features_list, action_taken=MCTSAction.ADD
        )

        # Mock simple performance history
        self.eval1 = FeatureEvaluation(
            feature_set=FeatureSet(features=[self.fd1], action_taken=MCTSAction.ADD),
            metrics=FeatureMetrics(
                extraction_time=0.1, missing_rate=0, unique_rate=0.5
            ),
            overall_score=0.5,
            fold_id="fold1",
            iteration=1,
            evaluation_time=1.0,
        )
        self.eval2 = FeatureEvaluation(
            feature_set=self.feature_set,  # Has f1 and f2
            metrics=FeatureMetrics(
                extraction_time=0.2, missing_rate=0, unique_rate=0.6
            ),
            overall_score=0.7,
            fold_id="fold1",
            iteration=2,
            evaluation_time=1.0,
        )
        self.performance_history: List[FeatureEvaluation] = [self.eval1, self.eval2]

        self.action_context = ActionContext(
            current_features=self.feature_set,
            performance_history=self.performance_history,
            available_actions=[MCTSAction.ADD, MCTSAction.MUTATE],
            max_features=3,
            max_cost=100,
        )

    def test_action_context_creation(self):
        self.assertEqual(len(self.action_context.current_features.features), 2)
        self.assertEqual(len(self.action_context.performance_history), 2)

    def test_can_add_feature(self):
        self.assertTrue(self.action_context.can_add_feature())
        self.action_context.max_features = 2
        self.assertFalse(self.action_context.can_add_feature())
        self.action_context.max_features = 1
        self.assertFalse(self.action_context.can_add_feature())

    def test_can_increase_cost(self):
        self.assertTrue(
            self.action_context.can_increase_cost(additional_cost=50)
        )  # 30 + 50 = 80 <= 100
        self.assertFalse(
            self.action_context.can_increase_cost(additional_cost=80)
        )  # 30 + 80 = 110 > 100
        self.action_context.max_cost = 25
        self.assertFalse(
            self.action_context.can_increase_cost(additional_cost=1)
        )  # 30 + 1 = 31 > 25

    def test_get_worst_performing_feature_empty_history(self):
        empty_history_context = ActionContext(
            current_features=self.feature_set,
            performance_history=[],
            available_actions=[MCTSAction.ADD],
            max_features=3,
        )
        self.assertIsNone(empty_history_context.get_worst_performing_feature())

    def test_get_worst_performing_feature_single_feature(self):
        single_feature_set = FeatureSet(
            features=[self.fd1], action_taken=MCTSAction.ADD
        )
        single_feature_context = ActionContext(
            current_features=single_feature_set,
            performance_history=self.performance_history,
            available_actions=[MCTSAction.ADD],
            max_features=3,
        )
        self.assertIsNone(single_feature_context.get_worst_performing_feature())

    def test_get_worst_performing_feature_logic(self):
        # This test depends on the _calculate_feature_performance_scores logic
        # With eval1 (f1 score 0.5) and eval2 (f1, f2 score 0.7),
        # f1's score should be: (0.5 from eval1) + (0.7-0.5)/2 from eval2 = 0.5 + 0.1 = 0.6
        # f2's score should be: (0.7-0.5) from being new in eval2 = 0.2
        # So f2 is considered "worse" because its score is lower by this logic.
        # Note: The logic penalizes removed features and rewards new features for improvement.
        # If features are common, improvement is distributed.

        # Create a history where f1 existed, then f2 was added and score improved
        f_set1 = FeatureSet(features=[self.fd1], action_taken=MCTSAction.ADD)
        f_set2 = FeatureSet(features=[self.fd1, self.fd2], action_taken=MCTSAction.ADD)

        eval_hist = [
            FeatureEvaluation(
                feature_set=f_set1,
                metrics=FeatureMetrics(extraction_time=0.1),
                overall_score=0.5,
                fold_id="f",
                iteration=1,
                evaluation_time=1,
            ),
            FeatureEvaluation(
                feature_set=f_set2,
                metrics=FeatureMetrics(extraction_time=0.1),
                overall_score=0.7,
                fold_id="f",
                iteration=2,
                evaluation_time=1,
            ),
        ]

        ac = ActionContext(
            current_features=f_set2,  # current has f1, f2
            performance_history=eval_hist,
            available_actions=[MCTSAction.REPLACE],
            max_features=5,
        )
        # Based on the logic in _calculate_feature_performance_scores:
        # f1 score: initially 0. Then (0.7 - 0.5) / 1 (common features in last step) = 0.2
        # f2 score: initially 0. Then (0.7 - 0.5) (new feature) = 0.2
        # In this case, they are equal. min() might pick the first one alphabetically or by insertion.
        # Let's adjust scores to make one clearly worse.

        f_set3 = FeatureSet(
            features=[self.fd2], action_taken=MCTSAction.ADD
        )  # f1 was removed, f2 remains
        eval_hist_2 = [
            FeatureEvaluation(
                feature_set=f_set1,
                metrics=FeatureMetrics(extraction_time=0.1),
                overall_score=0.8,
                fold_id="f",
                iteration=1,
                evaluation_time=1,
            ),  # f1 high
            FeatureEvaluation(
                feature_set=f_set2,
                metrics=FeatureMetrics(extraction_time=0.1),
                overall_score=0.6,
                fold_id="f",
                iteration=2,
                evaluation_time=1,
            ),  # f1,f2 - score dropped
            FeatureEvaluation(
                feature_set=f_set3,
                metrics=FeatureMetrics(extraction_time=0.1),
                overall_score=0.5,
                fold_id="f",
                iteration=3,
                evaluation_time=1,
            ),  # f2 - score dropped further
        ]
        ac_2 = ActionContext(
            current_features=f_set3,  # current has f2
            performance_history=eval_hist_2,
            available_actions=[MCTSAction.REPLACE],
            max_features=5,
        )
        # In ac_2 (current is f_set3 with only f2):
        # History:
        # 1: f1 (score 0.8)
        # 2: f1, f2 (score 0.6) -> f1 gets (0.6-0.8)/1 = -0.2, f2 gets (0.6-0.8) = -0.2
        # 3: f2 (score 0.5) -> f1 removed, its score is penalized by -(0.5-0.6) = +0.1. total -0.2 + 0.1 = -0.1
        #                      f2 is common, score gets (0.5-0.6)/1 = -0.1. total -0.2 - 0.1 = -0.3
        # This scenario is tricky as get_worst_performing_feature operates on `current_features`.
        # Let's test the direct output of _calculate_feature_performance_scores for clarity on current_features

        scores = self.action_context._calculate_feature_performance_scores()
        # self.action_context.current_features = {f1, f2}
        # History: eval1 (f1, score 0.5), eval2 (f1, f2, score 0.7)
        # f1: score_f1 = (0.7 - 0.5) / 1 (common feature in eval2 vs eval1) = 0.2
        # f2: score_f2 = (0.7 - 0.5) (new feature in eval2 vs eval1) = 0.2
        # They should be equal.
        self.assertIn(self.action_context.get_worst_performing_feature(), ["f1", "f2"])

    def test_get_best_performing_features(self):
        # Using the same logic as above, f1 and f2 have equal calculated scores of 0.2
        best_features = self.action_context.get_best_performing_features(top_k=1)
        self.assertEqual(len(best_features), 1)
        self.assertIn(best_features[0], ["f1", "f2"])

        best_features_k2 = self.action_context.get_best_performing_features(top_k=2)
        self.assertEqual(len(best_features_k2), 2)
        self.assertIn("f1", best_features_k2)
        self.assertIn("f2", best_features_k2)

    def test_get_feature_performance_summary(self):
        summary = self.action_context.get_feature_performance_summary()
        self.assertEqual(summary["total_features"], 2)
        self.assertEqual(summary["total_evaluations"], 2)
        self.assertIn(summary["best_feature"]["name"], ["f1", "f2"])
        self.assertIn(summary["worst_feature"]["name"], ["f1", "f2"])
        self.assertAlmostEqual(summary["feature_scores"]["f1"], 0.2)
        self.assertAlmostEqual(summary["feature_scores"]["f2"], 0.2)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
