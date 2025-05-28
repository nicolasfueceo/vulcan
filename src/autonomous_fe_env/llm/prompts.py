"""
Prompt templates for VULCAN LLM integration.

This module contains structured prompts for feature engineering and reflection tasks.
"""

from typing import Any, Dict, List


class FeatureEngineeringPrompts:
    """Prompt templates for feature engineering tasks."""

    @staticmethod
    def get_system_message() -> str:
        """Get the system message for feature engineering."""
        return """You are an expert feature engineer specializing in recommendation systems and machine learning. 

Your expertise includes:
- Deep understanding of collaborative filtering and content-based recommendation systems
- Advanced feature engineering techniques for user behavior analysis
- Temporal pattern recognition in user-item interactions
- Statistical and machine learning approaches to feature creation
- Experience with large-scale recommendation datasets like Goodreads, MovieLens, and Amazon

You generate high-quality, interpretable features that improve recommendation accuracy while maintaining computational efficiency."""

    @staticmethod
    def generate_feature_prompt(context: Dict[str, Any]) -> str:
        """Generate a comprehensive feature engineering prompt."""
        state_manager = context.get("state_manager")
        iteration = context.get("iteration", 1)

        current_features = []
        best_score = 0.0
        baseline_score = 0.42

        if state_manager:
            current_features = [f.name for f in state_manager.get_current_features()]
            best_score = state_manager.get_current_score()

        improvement = (
            ((best_score - baseline_score) / baseline_score * 100)
            if best_score > 0
            else 0
        )

        prompt = f"""# Feature Engineering Task for Goodreads Recommendation System

## Current Context
- **Iteration**: {iteration}
- **Best Score Achieved**: {best_score:.4f}
- **Baseline Score**: {baseline_score:.4f}
- **Current Improvement**: {improvement:.1f}%
- **Existing Features**: {current_features if current_features else "None (starting fresh)"}

## Dataset Schema
The Goodreads dataset contains the following key columns:
- `user_id`: Unique identifier for users
- `book_id`: Unique identifier for books  
- `rating`: User rating (1-5 stars)
- `review_text`: Text content of user reviews
- `date_added`: Timestamp when review was added
- `book_title`: Title of the book
- `author`: Book author name

## Available Data Access Patterns
Your feature function will receive:
1. `current_review_data`: The current user-book interaction being predicted
2. `horizontal_user_data`: All other reviews by the same user (user history)
3. `vertical_book_data`: All other reviews for the same book (book reviews)

## Feature Engineering Guidelines

### High-Impact Feature Categories:
1. **User Behavioral Patterns**
   - Rating consistency and variance
   - Temporal reading patterns
   - Genre exploration behavior
   - Review frequency and engagement

2. **Book Popularity Dynamics**
   - Recent vs historical popularity
   - Rating trends over time
   - Review velocity and momentum
   - Author popularity effects

3. **User-Book Interaction Features**
   - Similarity to user's historical preferences
   - Book's performance with similar users
   - Temporal context of the interaction
   - Cross-genre preference patterns

4. **Text-Based Features**
   - Review sentiment analysis
   - Review length and detail
   - Keyword and topic analysis
   - Writing style consistency

### Technical Requirements:
- Function name: `calculate_[feature_name]`
- Handle missing/empty data gracefully
- Return numeric values (float/int)
- Optimize for computational efficiency
- Include meaningful default values

## Task
Generate a novel feature that:
1. **Addresses a specific recommendation challenge**
2. **Leverages available data effectively**
3. **Has clear business interpretation**
4. **Differs from existing features**: {current_features}

## Response Format
Please provide your response in exactly this format:

```
Feature Name: [snake_case_name]
Description: [Clear explanation of what the feature measures and why it's valuable]
Implementation:
```python
def calculate_[feature_name](current_review_data, horizontal_user_data=None, vertical_book_data=None):
    # Your implementation here
    # Handle edge cases and missing data
    # Return a numeric value
    pass
```
Required Columns: [list of required columns]
Business Logic: [Explanation of the underlying hypothesis and expected impact]
```

## Innovation Focus
For iteration {iteration}, consider these strategic directions:
- If early iterations (1-5): Focus on fundamental user/item characteristics
- If mid iterations (6-15): Explore temporal patterns and interaction dynamics  
- If later iterations (16+): Consider complex feature interactions and ensemble approaches

Generate a feature that pushes the boundaries of recommendation system performance while maintaining interpretability."""

        return prompt

    @staticmethod
    def generate_feature_refinement_prompt(
        feature_name: str, performance: float, context: Dict[str, Any]
    ) -> str:
        """Generate a prompt for refining an existing feature."""
        return f"""# Feature Refinement Task

## Current Feature Analysis
- **Feature Name**: {feature_name}
- **Current Performance**: {performance:.4f}
- **Context**: {context}

## Refinement Objectives
1. Improve feature discriminative power
2. Handle edge cases better
3. Optimize computational efficiency
4. Enhance interpretability

Please suggest specific improvements to the feature implementation or propose a variant that addresses identified weaknesses."""


class ReflectionPrompts:
    """Prompt templates for reflection and strategic analysis."""

    @staticmethod
    def get_system_message() -> str:
        """Get the system message for reflection tasks."""
        return """You are a strategic AI research analyst specializing in machine learning and feature engineering optimization.

Your expertise includes:
- Performance analysis and trend identification
- Strategic planning for iterative improvement processes
- Pattern recognition in experimental results
- Recommendation system optimization strategies
- Research methodology and experimental design

You provide insightful analysis, identify patterns, and suggest strategic directions for continued improvement."""

    @staticmethod
    def generate_reflection_prompt(context: Dict[str, Any]) -> str:
        """Generate a comprehensive reflection prompt."""
        iteration = context.get("iteration", 1)
        current_score = context.get("current_score", 0.0)
        best_score = context.get("best_score", 0.0)
        feature_name = context.get("feature_name", "Unknown")
        recent_features = context.get("recent_features", [])
        performance_history = context.get("performance_history", [])

        prompt = f"""# Strategic Reflection and Analysis

## Current State Assessment
- **Current Iteration**: {iteration}
- **Latest Feature**: {feature_name}
- **Latest Score**: {current_score:.4f}
- **Best Score Achieved**: {best_score:.4f}
- **Recent Features**: {recent_features[-5:] if recent_features else "None"}

## Performance Analysis
{FeatureEngineeringPrompts._format_performance_history(performance_history)}

## Reflection Tasks

### 1. Performance Trend Analysis
Analyze the progression of feature performance:
- What patterns do you observe in the results?
- Are we seeing consistent improvement or plateauing?
- Which types of features are performing best?

### 2. Strategic Insights
Based on the current results:
- What does the latest feature performance tell us about the recommendation problem?
- What user or item characteristics seem most predictive?
- Are there unexplored areas with high potential?

### 3. Feature Engineering Strategy
For the next phase of development:
- What feature categories should we prioritize?
- Are there specific data patterns we should investigate?
- Should we focus on refinement or exploration?

### 4. Technical Recommendations
Provide specific guidance:
- Optimal feature complexity level
- Data utilization strategies
- Potential feature interaction opportunities

### 5. Risk Assessment
Identify potential challenges:
- Overfitting risks with current approach
- Data quality or availability issues
- Computational efficiency concerns

## Response Format
Provide a structured analysis addressing each reflection task with specific, actionable insights.

Focus on strategic direction rather than implementation details."""

        return prompt

    @staticmethod
    def generate_experiment_design_prompt(context: Dict[str, Any]) -> str:
        """Generate a prompt for experimental design and hypothesis formation."""
        return f"""# Experimental Design and Hypothesis Formation

## Current Context
{context}

## Task
Design the next set of experiments to systematically explore the feature space:

1. **Hypothesis Formation**: What specific hypotheses about user behavior or item characteristics should we test?

2. **Experimental Design**: How should we structure the next iterations to test these hypotheses effectively?

3. **Success Metrics**: What specific improvements would validate our hypotheses?

4. **Risk Mitigation**: How can we avoid common pitfalls in feature engineering?

Provide a structured experimental plan for the next 5-10 iterations."""

    @staticmethod
    def _format_performance_history(history: List[Dict[str, Any]]) -> str:
        """Format performance history for inclusion in prompts."""
        if not history:
            return "No performance history available."

        formatted = "Recent Performance History:\n"
        for i, entry in enumerate(history[-10:], 1):  # Last 10 entries
            score = entry.get("score", 0.0)
            feature = entry.get("feature_name", "Unknown")
            formatted += f"{i}. {feature}: {score:.4f}\n"

        return formatted
