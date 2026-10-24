# Tournament endpoint instructions

The round-robin tournament will run across sessions starting at **11:30 PM IST each night from Sunday, October 25 through Thursday, October 29, 2026**. Keep your computer awake, leave VS Code open, and keep the endpoint running during each session until the organizer confirms the tournament is over.

## Start your endpoint

Run these commands from the root of your assignment project, where `src/student/agent.py` is located. Use the same Python environment you use for the assignment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r tournament/requirements.txt
python tournament/server.py --agent src/student/agent.py --class Agent1 --port 5000
```

If the generated gRPC modules are missing, run `./generate.sh` from the project root before starting the endpoint.

The wrapper imports your `Agent1` from the required project structure and serves decisions on port `5000`. It uses the normal student interface and action policy. The wrapper prints a tournament token in the terminal when it starts. Keep that token private and send it to the organizer together with your forwarded URL.

## Make the port public in VS Code

1. Leave the server command running.
2. Open VS Code's **Ports** view. If port `5000` is not listed, choose **Forward a Port** from the Command Palette and enter `5000`.
3. Right-click the forwarded port and choose **Port Visibility → Public**.
4. Copy the **Forwarded Address**. Open `<forwarded-address>/health` in a browser. It should show `"status":"ok"` and `"protocol_version":1`.
5. Send the base forwarded address and the token to the organizer through the course's chosen private channel. Do not post the token publicly.

VS Code's built-in port forwarding supports making a forwarded port public. Public visibility means anyone with the URL can reach the service, so keep the token private. See the [VS Code port-forwarding guide](https://code.visualstudio.com/docs/debugtest/port-forwarding).

The forwarded URL and local port must remain the same and reachable for the tournament. If you restart the server or VS Code, check the address again and send the organizer the updated URL and token.

## Results and match replays

After the round robin finishes, the organizer will share the complete participant list, final standings, and all match logs. To visualize the shared `.rcg` logs on your machine, put them in a folder and run this from your assignment project root:

```bash
python a3.py watch --logs /path/to/match-logs
```

Open the local address printed by the command to browse the replays.

## Runtime expectations

The endpoint receives the simulator's full student-facing world model and returns the actions produced by `Agent1.on_ball` or `Agent1.off_ball`. The simulator enforces the normal action policy and decision limits. The network request is part of the one-second decision deadline, so keep the endpoint responsive. Each player number gets its own agent instance and memory for each match.
