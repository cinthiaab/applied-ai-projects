import unittest

import numpy as np
import pandas as pd

from experimento import ATALHO, ESTAVEL, MODELOS, gerar_dados, gerar_treinos, resumir, simular


class TestExperimento(unittest.TestCase):
    def test_inversao_preserva_sinal_estavel(self):
        X_f, y_f = gerar_dados(400, 1, np.random.default_rng(8))
        X_i, y_i = gerar_dados(400, -1, np.random.default_rng(8))
        np.testing.assert_array_equal(y_f, y_i)
        np.testing.assert_array_equal(X_f[:, ESTAVEL], X_i[:, ESTAVEL])
        np.testing.assert_allclose(X_f[:, ATALHO] - X_i[:, ATALHO], 4 * y_f)

    def test_treinos_tem_mesmo_tamanho_e_base_compartilhada(self):
        familiar, diverso = gerar_treinos(100, np.random.default_rng(8))
        self.assertEqual(familiar[0].shape, (100, 2))
        self.assertEqual(diverso[0].shape, (100, 2))
        np.testing.assert_array_equal(familiar[0][:50], diverso[0][:50])
        np.testing.assert_array_equal(familiar[1][:50], diverso[1][:50])

    def test_entradas_invalidas(self):
        with self.assertRaises(ValueError):
            gerar_dados(10, 2, np.random.default_rng(8))
        with self.assertRaises(ValueError):
            gerar_treinos(9, np.random.default_rng(8))
        with self.assertRaises(ValueError):
            simular([1, 1])

    def test_resultado_e_reproduzivel_e_completo(self):
        config = dict(repeticoes=4, n_treino=200, n_teste=100, semente=55)
        a = simular([1, -1], **config)
        pd.testing.assert_frame_equal(a, simular([1, -1], **config))
        self.assertEqual(len(a), 4 * 2 * len(MODELOS))
        self.assertFalse(a.duplicated(["repeticao", "ambiente", "modelo"]).any())
        self.assertTrue(a["acuracia"].between(0, 1).all())

    def test_ic_usa_variacao_entre_repeticoes(self):
        amostra = pd.DataFrame({"modelo": ["A", "A"], "acuracia": [0.8, 0.9]})
        resumo = resumir(amostra, ["modelo"]).iloc[0]
        self.assertAlmostEqual(resumo["media"], 0.85)
        self.assertAlmostEqual(resumo["margem"], 1.96 * 0.05)


if __name__ == "__main__":
    unittest.main()
