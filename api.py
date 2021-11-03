from fastapi  import FastAPI, Request


app = FastAPI()


@app.get("/client")
async def predict(request: Request):

    client_dict = await request.json()['json']

    return client_dict




     