async function loadData() {
  try {
    // on remonte d'un niveau, car data/ est à la racine
    const response = await fetch("../data/beartrap.json");
    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Erreur de chargement des données:", err);
    return { events: [] };
  }
}

function formatDamage(value) {
  if (value >= 1_000_000_000) {
    return (value / 1_000_000_000).toFixed(2) + "B";
  } else if (value >= 1_000_000) {
    return (value / 1_000_000).toFixed(2) + "M";
  } else if (value >= 1_000) {
    return (value / 1_000).toFixed(2) + "K";
  }
  return value.toString();
}

function updateLastEventInfo(event) {
  document.getElementById("last-event-name").textContent = event.name;
  document.getElementById("last-event-date").textContent = event.date;
  document.getElementById("last-event-players-count").textContent =
    event.players.length;

  const totalDamage = event.players.reduce(
    (sum, p) => sum + (p.damage || 0),
    0
  );
  document.getElementById("last-event-total-damage").textContent =
    formatDamage(totalDamage);
}

function updateRankingTable(event) {
  const tbody = document.querySelector("#ranking-table tbody");
  tbody.innerHTML = "";

  event.players
    .slice()
    .sort((a, b) => b.damage - a.damage)
    .forEach((player, index) => {
      const tr = document.createElement("tr");

      const rankTd = document.createElement("td");
      rankTd.textContent = index + 1;

      const nameTd = document.createElement("td");
      nameTd.textContent = player.name;

      const dmgTd = document.createElement("td");
      dmgTd.textContent = formatDamage(player.damage);

      tr.appendChild(rankTd);
      tr.appendChild(nameTd);
      tr.appendChild(dmgTd);

      tbody.appendChild(tr);
    });
}

function renderDamageChart(event) {
  const ctx = document.getElementById("damage-chart").getContext("2d");

  const sortedPlayers = event.players
    .slice()
    .sort((a, b) => b.damage - a.damage)
    .slice(0, 15); // top 15

  const labels = sortedPlayers.map((p) => p.name);
  const values = sortedPlayers.map((p) => p.damage);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Dégâts",
          data: values,
          backgroundColor: "rgba(75, 192, 192, 0.7)",
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: {
            color: "#f5f5f5",
          },
        },
        y: {
          ticks: {
            color: "#f5f5f5",
            callback: function (value) {
              return formatDamage(value);
            },
          },
        },
      },
    },
  });
}

function updateEventsHistory(events) {
  const ul = document.getElementById("events-list");
  ul.innerHTML = "";

  events
    .slice()
    .sort((a, b) => (a.date < b.date ? 1 : -1))
    .forEach((event) => {
      const li = document.createElement("li");
      li.textContent = `${event.date} – ${event.name} (${event.players.length} joueurs)`;
      ul.appendChild(li);
    });
}

async function init() {
  const data = await loadData();

  if (!data.events || data.events.length === 0) {
    console.warn("Aucun événement Bear Trap trouvé.");
    return;
  }

  const events = data.events;
  const lastEvent = events[events.length - 1];

  updateLastEventInfo(lastEvent);
  updateRankingTable(lastEvent);
  renderDamageChart(lastEvent);
  updateEventsHistory(events);
}

document.addEventListener("DOMContentLoaded", init);