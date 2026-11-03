
def node_consistency(csp, domains):
    """Remove domain values that violate unary constraints.

    Returns:
        consistent: bool
        domains: updated domains
        stats: statistics
    """

    updated_domains = {
        var: set(domain)
        for var, domain in domains.items()
    }

    for var in csp.variables:
        for constraint in csp.constraints:

            if len(constraint.scope) == 1 and var in constraint.scope:

                for node in updated_domains[var].copy():

                    if not constraint.is_satisfied({var: node}):
                        updated_domains[var].remove(node)

    consistent = all(
        len(domain) > 0
        for domain in updated_domains.values()
    )

    stats = {
        "values_removed": sum(
            len(domains[var])  - len(updated_domains[var])
            for var in csp.variables
        )
    }

    return consistent, updated_domains, stats