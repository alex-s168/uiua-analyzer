package me.alex_s168.uiua.mlir

fun function(
    name: String,
    args: List<Pair<MLIRVar, MLIRType>>,
    rets: List<Pair<MLIRVar, MLIRType>>,
    body: List<String>
): String {
    val res = StringBuilder()

    res.append("func.func @")
    res.append(name)
    res.append('(')
    args.forEachIndexed { index, (va, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(va)
        res.append(": ")
        res.append(ty)
    }
    res.append(") -> (")
    rets.forEachIndexed { index, (_, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(ty)
    }
    res.append(") {\n")

    body.forEach {
        res.append("  ")
        res.append(it)
        res.append('\n')
    }

    res.append(" return ")
    rets.forEachIndexed { index, (va, _) ->
        if (index > 0)
            res.append(", ")
        res.append(va)
    }
    res.append(" : ")
    rets.forEachIndexed { index, (_, ty) ->
        if (index > 0)
            res.append(", ")
        res.append(ty)
    }
    res.append("\n}")

    return res.toString()
}