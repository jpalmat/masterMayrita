(deftemplate A
	0 70
	(
		(ligero (0 1) (10 1) (20 0))
		(moderado (10 0) (20 1) (30 0))
		(severo (20 0) (30 1) (40 0))
		(muy_severo (30 0) (40 1) (70 1))
	)
)


(deftemplate B
	0 35
	(
		(debil (0 1) (5 1) (10 0))
		(moderada (5 0) (10 1) (15 0))
		(fuerte (10 0) (20 1) (30 0))
		(muy_fuerte (25 0) (30 1) (35 1))
	)
)

(deftemplate riesgo
	0 70
	(
		(bajo (0 1) (20 0))
		(medio (10 0) (30 1) (50 0))
		(alto (40 0) (60 1) (70 0))
	)
)





(defrule regla_1
	(A muy_severo)
	(B muy_fuerte)
=>
	(assert (riesgo alto))
)

(defrule regla_2
	(A moderado)
	(B muy_fuerte)
=>
	(assert (riesgo alto))
)

(defrule regla_3
	(A muy_severo)
	(B fuerte)
=>
	(assert (riesgo alto))
)

(defrule regla_4
	(A severo)
	(B fuerte)
=>
	(assert (riesgo medio))
)

(defrule regla_5
	(A muy_severo)
	(B moderada)
=>
	(assert (riesgo medio))
)

(defrule regla_6
	(A moderado)
	(B moderada)
=>
	(assert (riesgo medio))
)

(defrule regla_7
	(A ligero)
	(B moderada)
=>
	(assert (riesgo bajo))
)

(defrule regla_8
	(A muy_severo)
	(B debil)
=>
	(assert (riesgo medio))
)

(defrule regla_9
	(A moderado)
	(B debil)
=>
	(assert (riesgo bajo))
)

(defrule regla_10
	(A ligero)
	(B debil)
=>
	(assert (riesgo bajo))
)
