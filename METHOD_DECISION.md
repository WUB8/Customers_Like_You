# Method decision (ML1)

**Main method: Truncated SVD.** Online Retail II is naturally represented as a sparse customer-by-product matrix. Unlike ordinary PCA, Truncated SVD does not centre every product column. It preserves the sparse structure during fitting and directly supplies customer and product factor representations used for cosine-similarity recommendations. PCA is retained as an explicit comparison on the same training matrix and the same rank.

**Purchase value:** binary purchase indicator. A customer who orders 1,000 items on one invoice should not automatically overwhelm the signal from other customers. The optional ML7 experiment compares binary values with winsorized purchase counts and log(1 + count).

**Product catalogue:** items bought by 10 or more distinct customers in the **training period**; this reduces tail sparsity while avoiding test-set knowledge of product popularity. Only apparent product stock codes are included. Service/fee codes, cancelled invoices, nonpositive quantities/prices, and missing customer IDs are removed and counted in the log.

**Time split:** the final two months of transactions are held out. Only purchases of products that a test customer has never bought before, and that belong to the training catalogue, are eligible test targets. Customers not present during training cannot be assessed by this warm-start method. This limitation needs to be described to the stakeholder.

**Chosen rank:** k=50 is a pre-selected storage/compute compromise, not tuned on the held-out test set. Fit k in 10, 20, 50, 100 and 200 and inspect reconstruction error. Do not retroactively claim a test-optimal k if the ranking results were not used for a separate validation split.

**Recommendation rule:** L2-normalize each product's k-dimensional factor vector, add the cosine similarities between the candidate product and all products the customer previously purchased, exclude the products already purchased, and return the top 10. The same procedure is applied to the PCA factor vectors to make the ML5 comparison fair.

**Baseline:** for each eligible customer, suggest the ten most popular training products that customer has not previously purchased. Popularity is computed exclusively from training purchases. The baseline and the model share the same time-based held-out customer population.

**Key limitations and risks:**
- Discarding components throws away weaker purchase associations. Reconstruction error quantifies this loss on the training matrix.
- Ranking hit@10 measures only whether at least one future new item is in the list; it does not imply purchases were caused by recommendations.
- New customers/products need a fallback such as best sellers until they have enough history.
- Purchase logs may reflect availability and historical seasonality rather than stable preference.
- Data cleaning can remove legitimate stock codes; review the rejected codes if product coverage is a priority.
- Pure SVD may underperform best sellers; report the actual result, not a fabricated improvement.

See `results/cleaning_log.csv`, `results/svd_k_curve.csv`, `results/method_comparison.csv`, and `results/model_metrics.json` for measured results after running the scripts.
