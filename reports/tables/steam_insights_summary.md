# Steam Games Dataset — Resumen de Insights EDA

**Generado**: 2026-02-12
**Dataset**: `steam_games_features_2026-02-12.csv`
**Registros**: 122,610 juegos | 115 columnas de features

---

## KPIs del Dataset

| KPI | Valor Real |
|-----|-----------|
| Total de juegos | 122,610 |
| Rango de lanzamiento | 1997 - 2026 |
| Juegos gratuitos (is_free) | 21.4% |
| Precio mediano (juegos de pago) | USD 3.49 |
| Mediana de reviews positivas (raw) | 5 |
| Juegos con score Metacritic | 3.5% |
| Soporte Windows | 100.0% |
| Soporte Mac | 17.4% |
| Soporte Linux | 12.8% |
| Ratio positivo mediano | 81.8% |

---

## Top 10 Hallazgos del EDA

| # | Hallazgo | Métrica Real | Implicación |
|---|----------|-------------|-------------|
| 1 | Dominancia moderada de F2P | 21.4% del catálogo es free-to-play | Separar análisis F2P vs. pago; no es mayoría sino minoría significativa |
| 2 | Era "Actual" concentra lanzamientos | 71,968 juegos desde 2022 (58.7% del total) | El catálogo crece exponencialmente; muchos son juegos de corta vida |
| 3 | Correlación reviews-ratio débil | r = 0.019 entre positive_ratio y total_reviews | Ratio de aprobación es independiente del volumen; alta varianza en juegos pequeños |
| 4 | Cobertura limitada de Metacritic | Solo 3.5% de juegos tienen score Metacritic | Variable auxiliar; inviable como target principal por falta de datos |
| 5 | Alto ratio positivo general | Mediana positive_ratio = 81.8% | Los usuarios de Steam tienden fuertemente a dejar reviews positivas |
| 6 | Action domina el catálogo | Action: 37.4% (45,888 juegos) del catálogo | Mercado saturado en Action; nichos como Strategy tienen mejor valoración Metacritic |
| 7 | Windows = soporte universal | 100% de juegos soportan Windows | Sin varianza; eliminar esta feature de modelos predictivos |
| 8 | Precios muy concentrados bajo $20 | 97.4% de juegos de pago cuestan ≤ $20; p75 = $6.99 | Distribución extremadamente sesgada; log-transform obligatorio |
| 9 | Long tail de developers | 76.5% de developers tienen solo 1 juego en Steam | Mercado ultrafragmentado; usar frequency encoding para developer |
| 10 | Playtime mediano es cero | F2P: 0 min mediano, Pago: 0 min mediano | La mayoría de juegos no tienen datos de playtime; feature con alta esparsidad |

---

## Estadísticas por Género (Top 10 por volumen)

| Género | Total Juegos | Precio Mediano | Ratio Positivo Med. | Metacritic Prom. | Edad Prom. (años) |
|--------|-------------|---------------|---------------------|------------------|-------------------|
| Action | 45,888 | $2.99 | 81.2% | 3.4 | 4.3 |
| Adventure | 24,053 | $2.99 | 83.5% | 3.1 | 4.1 |
| Casual | 23,826 | $1.99 | 84.1% | 0.5 | 3.7 |
| Indie | 10,723 | $2.99 | 81.4% | 2.3 | 4.1 |
| Unknown | 8,412 | $0.00 | 77.3% | 0.07 | 1.9 |
| Simulation | 2,449 | $4.99 | 75.0% | 4.4 | 4.5 |
| RPG | 1,910 | $4.99 | 81.5% | 6.4 | 4.2 |
| Strategy | 1,585 | $3.99 | 80.9% | 9.9 | 5.5 |
| Free To Play | 870 | $0.00 | 73.4% | 0.5 | 5.1 |
| Racing | 549 | $3.99 | 78.2% | 5.9 | 5.3 |

**Nota**: Strategy tiene el Metacritic promedio más alto (9.9) pero solo 1,585 títulos.

---

## Estadísticas por Era de Lanzamiento

| Era | Total Juegos | Precio Mediano | Ratio Positivo Med. | Años |
|-----|-------------|---------------|---------------------|------|
| Actual (2022+) | 71,968 | $2.39 | 87.5% | 2022-2026 |
| Explosion (2018-2021) | 34,574 | $1.99 | 79.9% | 2018-2021 |
| Direct Era (2014-2017) | 14,098 | $2.49 | 74.3% | 2014-2017 |
| Greenlight Era (2010-2013) | 1,304 | $2.49 | 79.2% | 2010-2013 |
| Pre-Greenlight (< 2010) | 666 | $2.49 | 83.8% | 1997-2009 |

**Observación clave**: La era "Actual" tiene el ratio positivo más alto (87.5%), posiblemente por sesgo de muestra pequeña en juegos recientes con pocas reviews.

---

## Recomendaciones para Modelado

| Modelo | Target | Features Clave | Baseline Esperado |
|--------|--------|---------------|-------------------|
| Random Forest Regressor | positive_ratio | genre_primary, game_age_years, price, dlc_count, log_total_reviews | MAE < 0.10, R2 > 0.50 |
| XGBoost Regressor | log_total_reviews (popularidad) | genre_primary, price_tier, era, is_free, game_age_years | RMSE < 1.5, R2 > 0.60 |
| Logistic Regression | is_free (binario) | genre_primary, developer_freq, publisher_freq, n_platforms | AUC-ROC > 0.85, F1 > 0.80 |
| LightGBM Classifier | price_tier (multiclase) | genre_primary, era, developer_freq, metacritic_score, n_platforms | Accuracy > 0.70, Macro-F1 > 0.60 |
| KMeans Clustering | Segmentación (no supervisado) | price, positive_ratio, log_total_reviews, game_age_years, genre_encoded | Silhouette Score > 0.35 |

---

## Advertencias y Consideraciones Técnicas

1. **Windows** tiene varianza cero (100% soporte) — excluir de modelos predictivos.
2. **average_playtime_forever** tiene mediana = 0 — alta esparsidad; tratar como feature opcional.
3. **metacritic_score** disponible solo para 3.5% — imputar con 0 o usar indicador binario `has_metacritic`.
4. **positive_ratio** puede ser inestable en juegos con menos de 10 reviews — filtrar o usar weight by reviews.
5. **developer_primary** tiene 76.5% de desarrolladores con un solo juego — usar `developer_freq` (frequency encoding).
6. **La era "Actual"** puede estar sobrerepresentada por juegos sin reviews suficientes — aplicar filtros de calidad mínima al modelar.

---

## Inventario de Archivos del Proyecto

| Tipo | Archivo |
|------|---------|
| Dataset limpio (Parquet) | `data/processed/steam_games_clean_2026-02-11.parquet` |
| Dataset features (CSV) | `data/processed/steam_games_features_2026-02-12.csv` |
| Notebook Limpieza | `notebooks/exploratory/01_limpieza_exploracion_steam.ipynb` |
| Notebook Features | `notebooks/exploratory/02_transformaciones_features_steam.ipynb` |
| Notebook Visualizaciones | `notebooks/exploratory/03_visualizaciones_insights_steam.ipynb` |
| Reporte Final | `notebooks/reports/04_reporte_final_steam.ipynb` |
| Figuras EDA (12 plots) | `reports/figures/viz1_*.png` a `viz12_*.png` |
| Figuras Reporte (3 plots) | `reports/figures/reporte_distribuciones.png`, `reporte_correlaciones.png`, `reporte_precio_tier.png` |

---

*Generado automáticamente el 2026-02-12 como parte del pipeline EDA del proyecto Gestion_Datos.*
