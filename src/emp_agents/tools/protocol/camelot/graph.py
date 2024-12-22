import httpx

GRAPH_URL = "https://thegraph.com/explorer/api/playground/QmbeQwnRSX4Fs9Bo3LZCAvkb4psdmcS8AqKv78Tcq6ohED"


async def get_pairs_by_volume(limit: int = 5) -> list[dict]:
    def map_response(response: dict) -> dict:
        return {
            "id": response["id"],
            "token0Address": response["token0"]["id"],
            "token0Symbol": response["token0"]["symbol"],
            "token1Address": response["token1"]["id"],
            "token1Symbol": response["token1"]["symbol"],
            "fee": response["feeZtO"],
            "volume": response["volumeUSD"],
        }

    query = (
        """
        {
            pools(first:%s, orderBy:volumeUSD, orderDirection:desc) {
                id
                token0 {
                    id
                    symbol
                }
                token1 {
                    id
                    symbol
                }
                feeZtO
                volumeUSD
            }
        }
    """
        % limit
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(GRAPH_URL, json={"query": query})
    return [map_response(pool) for pool in response.json()["data"]["pools"]]


async def get_pairs_for_token(token_address: str) -> list[dict]:
    def map_response(response: dict) -> dict:
        return {
            "id": response["id"],
            "token0Address": response["token0"]["id"],
            "token0Symbol": response["token0"]["symbol"],
            "token1Address": response["token1"]["id"],
            "token1Symbol": response["token1"]["symbol"],
            "fee": response["feeZtO"],
            "volume": response["volumeUSD"],
        }

    query = """
        {
            pools(
                first: 5,
                orderBy: volumeUSD,
                orderDirection:desc,
                where: {
                or: [
                    { token0: "%s" },
                    { token1: "%s" }
                ]
                }
            ) {
                id
                token0 {
                id
                symbol
                }
                token1 {
                id
                symbol
                }
                feeZtO
                volumeUSD
            }
        }
    """ % (
        token_address,
        token_address,
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(GRAPH_URL, json={"query": query})
    return [map_response(pool) for pool in response.json()["data"]["pools"]]


async def get_tokens_for_symbol(symbol: str) -> list[dict]:
    def map_response(response: dict) -> dict:
        return {
            "address": response["id"],
            "name": response["name"],
            "symbol": response["symbol"],
        }

    query = """
        {
            tokens(where:{
            symbol:"ARB"
        }){
                id
                name
                symbol
            }
        }
    """

    async with httpx.AsyncClient() as client:
        response = await client.post(GRAPH_URL, json={"query": query})
    return [map_response(token) for token in response.json()["data"]["tokens"]]
